"""
Runs one source end to end: read frames -> inference (every Nth frame) ->
geometry gate -> animal veto -> flicker filter -> stillness ->
draw -> save frame -> clip -> log event
-> optionally show a live window.

One SourcePipeline instance = one camera / one video / one folder. Kept
independent on purpose: a bad RTSP stream or a corrupt file in one source
should not stop the others (mirrors the per-camera isolation in the OWLv2
project this is a sibling of).
"""

import os
import time

import cv2

from .audit import AuditLogger
from .clip_recorder import ClipRecorder
from .events import EventLogger
from .temporal import StillnessGate, TemporalConfirmer


class SourcePipeline:
    def __init__(self, source, detector, cfg, allow_window, frame_sink=None,
                 veto=None, source_cfg_mount="oblique"):
        self.source = source
        self.detector = detector
        self.cfg = cfg
        self.source_id = source.source_id
        self.frame_skip = max(1, cfg["processing"]["frame_skip"])
        # Mount geometry is CATEGORICAL, not a tuned number: "do not tune per
        # camera" is right for scalar thresholds and wrong for how the camera is
        # bolted to the building. An enum survives a 20-camera rollout.
        #
        # min_aspect_ratio is an AXIS-ALIGNED test. On an oblique mount gravity
        # plus viewing angle keep a standing body roughly vertical in the image
        # and a fallen one roughly horizontal, so w/h separates them. Overhead,
        # gravity constrains nothing in the image plane and body orientation is
        # uniform over 360 degrees: a standing worker facing along image-x reads
        # w/h about 1.8 and PASSES, while someone lying along image-y reads
        # about 0.26 and is REJECTED. Both reported field symptoms are that one
        # defect, and it is deterministic on body orientation.
        #
        # There is no rotation-invariant replacement in this pipeline yet, so
        # overhead mounts switch the test off rather than run it backwards.
        # Deleting real falls is worse than admitting upright people, and the
        # audit log now records what that costs.
        self.mount = source_cfg_mount
        if self.mount == "overhead":
            self.min_aspect_ratio = 0.0          # axis-aligned test is invalid here
            self.edge_mode = "none"              # BUG 7: nobody walks into an
                                                 # overhead lens, so the rule only
                                                 # deletes falls near the border
        else:
            self.min_aspect_ratio = cfg["geometry"]["min_aspect_ratio"]
            self.edge_mode = cfg["geometry"]["reject_edge_touching"]
        self.edge_margin = cfg["geometry"]["edge_margin_px"]
        self.veto = veto
        self.conf_threshold = cfg["prompts"]["conf_threshold"]

        t_cfg = cfg["temporal"]
        self._confirmer = TemporalConfirmer(
            t_cfg["confirm_frames"], t_cfg["window_frames"],
        ) if t_cfg["enabled"] else None

        s_cfg = cfg["stillness"]
        # seconds -> INFERENCE frames: the gate only sees frames we ran inference on
        fps = getattr(source, "fps_hint", lambda: 25.0)()
        still_frames = max(1, round(s_cfg["still_seconds"] * fps / self.frame_skip))
        self.stillness = StillnessGate(
            still_frames, s_cfg["max_drift"], s_cfg["match_iou"],
            s_cfg["max_growth"],
            max(1, round(s_cfg["track_timeout_seconds"] * fps / self.frame_skip)),
            s_cfg["min_observations"],
        ) if s_cfg["enabled"] else None
        if self.stillness is not None:
            print(f"[{self.source_id}] stillness: a box must hold position for "
                  f"{still_frames} inference frames (~{s_cfg['still_seconds']:.1f}s "
                  f"at {fps:.0f}fps / skip {self.frame_skip}) before it alarms")


        out_cfg = cfg["output"]
        base = out_cfg["base_folder"]
        self.frame_dir = os.path.join(base, "frames", self.source_id)
        self.clip_dir = os.path.join(base, "clips", self.source_id)
        os.makedirs(self.frame_dir, exist_ok=True)
        os.makedirs(self.clip_dir, exist_ok=True)

        self.save_frames = out_cfg["save_frames"]
        self.save_clips = out_cfg["save_clips"]
        self.frame_min_interval = out_cfg["frame_min_interval_seconds"]
        self._last_save_time = 0.0

        # frame_sink set -> this pipeline is one tile of a multi-camera window
        # driven by the main thread; it must never touch cv2.imshow itself.
        self.frame_sink = frame_sink
        self.show_window = (allow_window and cfg["display"]["show_window"]
                            and frame_sink is None)
        self.window_name = f"{cfg['display']['window_name']} | {self.source_id}"

        self.events = EventLogger(cfg["logging"]["events_path"])
        self.audit = AuditLogger(cfg["logging"].get("audit_path"),
                                 cfg["logging"].get("audit_enabled", True))

        self.show_candidates = cfg["display"]["show_candidates"]

        # BUG 12: a safety system that has silently stopped watching is more
        # dangerous than one with known misses. Nothing else in the pipeline
        # notices a dead stream, an RTSP reconnect loop, or a stalled thread.
        self.stale_after = cfg["logging"].get("stale_after_seconds", 30.0)
        self._last_frame_wall = time.time()
        self._stale_warned = False

        self._alarm = False
        self._draw_positives = []
        self._draw_candidates = []

        self._recorder = None  # created lazily once we know real frame size
        self._fps_hint = getattr(source, "fps_hint", lambda: 25.0)()
        self._pre_seconds = out_cfg["clip_pre_seconds"]
        self._post_seconds = out_cfg["clip_post_seconds"]

        self._stop = False


    def run(self):
        detection_count = 0
        frame_count = 0

        for frame_number, video_time_sec, frame in self.source:
            now = time.time()
            if now - self._last_frame_wall > self.stale_after:
                gap = now - self._last_frame_wall
                print(f"[{self.source_id}] WATCHDOG: no frame for {gap:.0f}s "
                      f"(limit {self.stale_after:.0f}s) -- this source was not "
                      f"watching anything for that period")
                self.audit.log(self.source_id, frame_number, video_time_sec, None,
                               "watchdog", "stalled", reason="frame_gap",
                               measured=gap, limit=self.stale_after)
            self._last_frame_wall = now

            if self._stop:
                break
            frame_count += 1

            if self._recorder is None and self.save_clips:
                h, w = frame.shape[:2]
                self._recorder = ClipRecorder(
                    self.clip_dir, self.source_id, self._fps_hint, (w, h),
                    self._pre_seconds, self._post_seconds,
                )

            run_inference = (frame_number % self.frame_skip == 0)
            positives = []

            if run_inference:
                A = self.audit
                fn, vt = frame_number, video_time_sec

                positives, shadows = self.detector.infer(frame)
                # a distractor class swallowed a region with no positive box
                # near it -- possible silent loss of a real casualty
                for sh in shadows:
                    A.log(self.source_id, fn, vt, sh, "detector", "rejected",
                          reason="negative_class_shadow", measured=sh.conf)

                before = positives
                positives = [p for p in positives
                             if p.conf >= self.conf_threshold]
                for p in before:
                    if p.conf < self.conf_threshold:
                        A.log(self.source_id, fn, vt, p, "conf", "rejected",
                              reason="below_threshold", measured=p.conf,
                              limit=self.conf_threshold)

                # geometry gate before stillness: a standing person scoring 0.55
                # must never reach the countdown, or it waits it out as readily
                # as a real one
                positives, geo_rej = self.detector.filter_by_geometry(
                    positives, self.min_aspect_ratio, frame.shape,
                    self.edge_mode, self.edge_margin,
                )
                for p, reason, measured, limit in geo_rej:
                    A.log(self.source_id, fn, vt, p, "geometry", "rejected",
                          reason=reason, measured=measured, limit=limit)

                # veto pass runs only when something already fired, and before
                # stillness, so an animal never reaches the gate
                if self.veto is not None and positives:
                    if hasattr(self.detector, "apply_veto"):
                        positives, vetoed = self.detector.apply_veto(frame, positives)
                    else:
                        positives, vetoed = self.veto.apply(frame, positives)
                    for v in (vetoed or []):
                        det = v[0] if isinstance(v, tuple) else v
                        A.log(self.source_id, fn, vt, det, "veto", "rejected",
                              reason="animal", measured=getattr(det, "conf", None))

                # everything that survived the filters but has not yet earned an
                # alarm. Drawn yellow so the pipeline is visible rather than
                # silently discarding boxes.
                candidates = list(positives)

                # flicker filter BEFORE stillness: a one-frame blip should not
                # start a stillness track at all, or it occupies one for the
                # whole timeout while proving nothing
                if self._confirmer is not None:
                    pre = positives
                    positives = self._confirmer.update(positives)
                    if pre and not positives:
                        for p in pre:
                            A.log(self.source_id, fn, vt, p, "flicker", "rejected",
                                  reason="awaiting_repeat_evidence",
                                  measured=sum(self._confirmer._window),
                                  limit=self._confirmer.confirm_frames)

                # stillness last: a crouching person scores as high as a fallen
                # one, so only time separates them
                if self.stillness is not None:
                    pre = positives
                    positives = self.stillness.update(positives)
                    held = [p for p in pre if p not in positives]
                    for p in held:
                        info = self.stillness.track_info(p.xyxy)
                        if info is None:
                            A.log(self.source_id, fn, vt, p, "stillness", "rejected",
                                  reason="new_track")
                        else:
                            md = info["median_drift"]
                            moving = md is not None and md > info["max_drift"]
                            A.log(self.source_id, fn, vt, p, "stillness", "rejected",
                                  reason="moving" if moving else "accumulating",
                                  measured=(md if moving else info["elapsed"]),
                                  limit=(info["max_drift"] if moving else info["need"]))

                # a surviving detection IS the alarm. Nothing downstream keeps
                # an alarm alive on a frame the model did not fire on.
                alarm = len(positives) > 0

                self._alarm = alarm
                self._draw_positives = positives
                confirmed_boxes = {tuple(d.xyxy) for d in positives} if alarm else set()
                self._draw_candidates = [c for c in candidates
                                         if tuple(c.xyxy) not in confirmed_boxes]

                if self.show_candidates:
                    for det in self._draw_candidates:
                        self._draw_candidate(frame, det)

                if alarm:
                    for det in positives:
                        self._draw(frame, det)
                        detection_count += 1
                        self.events.log(self.source_id, frame_number,
                                        video_time_sec, det)
                        A.log(self.source_id, fn, vt, det, "alarm", "alarm")
                    # once per frame, not once per detection -- the saved image
                    # is the whole frame, so a second write for a second box
                    # would just overwrite the same picture
                    self._maybe_save_frame(frame, positives, video_time_sec)
            else:
                # skipped frame: redraw the current state so boxes do not strobe
                # at the frame_skip rate
                if self.show_candidates:
                    for det in getattr(self, "_draw_candidates", []):
                        self._draw_candidate(frame, det)
                if getattr(self, "_alarm", False):
                    for det in self._draw_positives:
                        self._draw(frame, det)

            detected_this_frame = getattr(self, "_alarm", False)
            if self._recorder is not None:
                # returns the closed clip's path, which ClipRecorder already
                # prints; nothing further to do with it here
                self._recorder.add_frame(frame, detected_this_frame)

            # count the state actually on screen, not this iteration's local
            # list -- on a frame_skip frame that list is empty even while an
            # alarm is up, which made the status bar contradict the boxes
            n_confirmed = len(self._draw_positives) if self._alarm else 0
            self._draw_status(frame, video_time_sec, frame_number, n_confirmed)

            if self.frame_sink is not None:
                self.frame_sink(self.source_id, frame)

            if self.show_window:
                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print(f"[{self.source_id}] stopped by user")
                    break
                if key == ord(" "):
                    print(f"[{self.source_id}] paused - press SPACE to continue")
                    while True:
                        k2 = cv2.waitKey(0) & 0xFF
                        if k2 == ord(" "):
                            break
                        if k2 == ord("q"):
                            self._stop = True
                            break

        if self._recorder is not None:
            self._recorder.close()
        if self.show_window:
            cv2.destroyWindow(self.window_name)

        print(f"[{self.source_id}] finished. frames={frame_count} detections={detection_count}")
        return detection_count

    def _draw(self, frame, det):
        x1, y1, x2, y2 = det.xyxy
        colour = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        label = f"MAN DOWN {det.conf:.2f}"
        cv2.rectangle(frame, (x1, max(0, y1 - 35)), (x1 + 300, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    def _draw_candidate(self, frame, det):
        """A box that passed the filters but has not been confirmed. Yellow, with
        the numbers that decide its fate, so a miss can be diagnosed from the
        video instead of guessed at."""
        x1, y1, x2, y2 = det.xyxy
        h, w = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        colour = (0, 220, 220)                       # BGR yellow
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        lines = [f"cand {det.conf:.2f}  w/h {bw/max(1,bh):.2f}  "
                 f"area {100.0*bw*bh/(w*h):.2f}%"]
        info = self.stillness.track_info(det.xyxy) if self.stillness else None
        if info is None:
            lines.append("still: no track yet")
        else:
            md = info["median_drift"]
            lines.append(
                f"still {info['elapsed']}/{info['need']}f  "
                f"obs {info['observations']}/{info['min_observations']}  "
                f"drift {('-' if md is None else f'{md:.4f}')}/{info['max_drift']}")

        y = max(14, y1 - 6 - 16 * len(lines))
        for i, text in enumerate(lines):
            ty = y + i * 16
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, ty - th - 3), (x1 + tw + 6, ty + 3), (0, 0, 0), -1)
            cv2.putText(frame, text, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, colour, 1)

    def _draw_status(self, frame, video_time_sec, frame_number, n_detections):
        n_cand = len(getattr(self, "_draw_candidates", []))
        status = (f"{self.source_id} | t={video_time_sec:.1f}s | frame={frame_number} "
                  f"| CONFIRMED={n_detections} | candidates={n_cand}")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, status, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    def _maybe_save_frame(self, frame, detections, video_time_sec):
        """Save the whole annotated frame.

        Called after _draw, so the boxes and labels are already burnt in -- the
        point of keeping the full frame rather than the crop is to see where in
        the scene the person is and what is around them, which a tight crop
        throws away.
        """
        if not self.save_frames or not detections:
            return
        now = time.time()
        if now - self._last_save_time < self.frame_min_interval:
            return
        self._last_save_time = now

        best = max(detections, key=lambda d: d.conf)
        filename = (
            f"{self.source_id}_t{video_time_sec:.2f}s_conf{best.conf:.2f}.jpg"
        )
        cv2.imwrite(os.path.join(self.frame_dir, filename), frame)
