"""
Runs one source end to end: read frames -> inference (every Nth frame) ->
geometry gate -> animal veto -> stillness -> temporal confirm ->
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

from .clip_recorder import ClipRecorder
from .events import EventLogger
from .temporal import StillnessGate, TemporalConfirmer


class SourcePipeline:
    def __init__(self, source, detector, cfg, allow_window, frame_sink=None,
                 veto=None):
        self.source = source
        self.detector = detector
        self.cfg = cfg
        self.source_id = source.source_id
        self.frame_skip = max(1, cfg["processing"]["frame_skip"])
        self.min_aspect_ratio = cfg["geometry"]["min_aspect_ratio"]
        self.edge_mode = cfg["geometry"]["reject_edge_touching"]
        self.edge_margin = cfg["geometry"]["edge_margin_px"]
        self.veto = veto
        self.conf_threshold = cfg["prompts"]["conf_threshold"]

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

        t_cfg = cfg["temporal"]
        self.temporal_enabled = t_cfg["enabled"]
        self._confirmer = TemporalConfirmer(
            t_cfg["confirm_frames"], t_cfg["window_frames"], t_cfg["hold_frames"],
        ) if self.temporal_enabled else None

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

        self.show_candidates = cfg["display"]["show_candidates"]

        self._alarm = False
        self._draw_positives = []
        self._draw_candidates = []
        self._holding = False

        self._recorder = None  # created lazily once we know real frame size
        self._fps_hint = getattr(source, "fps_hint", lambda: 25.0)()
        self._pre_seconds = out_cfg["clip_pre_seconds"]
        self._post_seconds = out_cfg["clip_post_seconds"]

        self._stop = False


    def run(self):
        detection_count = 0
        frame_count = 0

        for frame_number, video_time_sec, frame in self.source:
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
                positives = self.detector.infer(frame)
                # geometry gate before temporal: a standing person scoring 0.55
                # must never reach the confirmer, or it confirms just as readily
                # as a real one
                positives = self.detector.filter_by_geometry(
                    positives, self.min_aspect_ratio, frame.shape,
                    self.edge_mode, self.edge_margin,
                )
                positives = [p for p in positives
                             if p.conf >= self.conf_threshold]
                # veto pass runs only when something already fired, and before
                # temporal, so an animal never reaches the confirmer
                if self.veto is not None and positives:
                    if hasattr(self.detector, "apply_veto"):
                        positives, _ = self.detector.apply_veto(frame, positives)
                    else:
                        positives, _ = self.veto.apply(frame, positives)



                # everything that survived the filters but has not yet earned an
                # alarm. Drawn yellow so the pipeline is visible rather than
                # silently discarding boxes.
                candidates = list(positives)

                # stillness AFTER the veto and BEFORE the confirmer: a crouching
                # person scores as high as a fallen one, so only time separates
                # them, and the confirmer must never see an unproven box
                if self.stillness is not None:
                    positives = self.stillness.update(positives)

                if self._confirmer is not None:
                    alarm, positives = self._confirmer.update(positives)
                    holding = self._confirmer.holding
                else:
                    alarm, holding = len(positives) > 0, False

                self._alarm = alarm
                self._draw_positives = positives
                self._holding = holding
                confirmed_boxes = {tuple(d.xyxy) for d in positives} if alarm else set()
                self._draw_candidates = [c for c in candidates
                                         if tuple(c.xyxy) not in confirmed_boxes]

                if self.show_candidates:
                    for det in self._draw_candidates:
                        self._draw_candidate(frame, det)

                if alarm:
                    for det in positives:
                        self._draw(frame, det, dim=holding)
                    # only log/save on live detections -- a held frame is an
                    # inference we never ran, not evidence
                    if not holding:
                        for det in positives:
                            detection_count += 1
                            self.events.log(self.source_id, frame_number,
                                            video_time_sec, det)
                        # once per frame, not once per detection -- the saved
                        # image is the whole frame, so a second write for a
                        # second box would just overwrite the same picture
                        self._maybe_save_frame(frame, positives, video_time_sec)
            else:
                # skipped frame: redraw the current state so boxes do not strobe
                # at the frame_skip rate
                if self.show_candidates:
                    for det in getattr(self, "_draw_candidates", []):
                        self._draw_candidate(frame, det)
                if getattr(self, "_alarm", False):
                    for det in self._draw_positives:
                        self._draw(frame, det, dim=self._holding)

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

    def _draw(self, frame, det, dim=False):
        x1, y1, x2, y2 = det.xyxy
        colour = (0, 120, 200) if dim else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        label = f"MAN DOWN {det.conf:.2f}" + (" (hold)" if dim else "")
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
