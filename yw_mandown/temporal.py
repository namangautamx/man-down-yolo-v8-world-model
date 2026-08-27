"""
Temporal confirmation + hold, so a man-down alarm is a state rather than a
per-frame coin flip.

Measured on the cam1 live run before this existed: 158 frames carried a
detection, spread over 29 separate runs, 13 of them a single frame long, with
only 82% of consecutive detections actually adjacent. The person never moved
during those gaps -- the model just lost them for a frame or two. That is what
"not continuous" looks like on screen.

Two knobs, doing two different jobs:

  confirm_frames / window_frames  raise the alarm only after N of the last M
                                  inference frames detected something. Kills
                                  the 13 single-frame blips, which is also
                                  where most brief false positives live.

  hold_frames                     once raised, keep the alarm up for this many
                                  consecutive empty inference frames before
                                  dropping it. Bridges the gaps, so one fall is
                                  one continuous alarm and one clip instead of
                                  five fragments.

Counted in INFERENCE frames, not video frames: with processing.frame_skip: 2,
hold_frames: 15 is 30 video frames.
"""

import math

from collections import deque

from .detector import iou


class TemporalConfirmer:
    def __init__(self, confirm_frames, window_frames, hold_frames):
        if window_frames < confirm_frames:
            raise ValueError("window_frames must be >= confirm_frames")
        self.confirm_frames = confirm_frames
        self.hold_frames = hold_frames
        self._window = deque(maxlen=window_frames)
        self._active = False
        self._empty_streak = 0
        self._last_detections = []

    @property
    def active(self):
        return self._active

    @property
    def holding(self):
        """True when the alarm is up on hold rather than a live detection --
        the pipeline draws these boxes dimmer so what is real stays visible."""
        return self._active and self._empty_streak > 0

    def update(self, detections):
        """Feed one inference frame. Returns (alarm_active, detections_to_draw).

        detections_to_draw is the live set when there is one, otherwise the last
        live set while holding -- without it the box vanishes mid-hold and the
        display flickers exactly as before, even though the alarm stayed up.
        """
        had = len(detections) > 0
        self._window.append(had)

        if had:
            self._empty_streak = 0
            self._last_detections = list(detections)
            if not self._active and sum(self._window) >= self.confirm_frames:
                self._active = True
        else:
            self._empty_streak += 1
            if self._active and self._empty_streak > self.hold_frames:
                self._active = False
                self._last_detections = []

        if not self._active:
            return False, []
        return True, (list(detections) if had else list(self._last_detections))



def _median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


class StillnessGate:
    """Hold a detection back until it has proved it is not moving.

    Why this exists
    ---------------
    Of 19 false positives that survived every other filter, 14 were people who
    are genuinely not down: crouching for a bag, kneeling by a car, bending at a
    desk, sitting on a sofa. They scored 0.67, 0.70, 0.84. From a single box,
    crouching and lying are not separable, so no prompt or threshold reaches
    them. Time does -- someone crouching is gone in seconds, someone who has
    collapsed is not.

    Measured on cam2 clips, drift per inference frame in units of box size:

        genuinely fallen   0.0009  0.0012  0.0013  0.0015  0.0046
        walking / active   0.0276  0.0320  0.0373

    Thirty times apart, normalised by box size so it is independent of
    resolution and distance. Growth is checked too: a person walking INTO the
    lens grew 4-7% of box area per frame, a fallen person 0.2%.

    THIS DELAYS EVERY ALARM by still_seconds. That is the mechanism, not a side
    effect. Paid once per event: once a track is confirmed it stays confirmed.

    Two mistakes this went through, both worth recording
    -----------------------------------------------------
    1. Demanding N CONSECUTIVE still frames lost all 8 lab falls. A fallen
       person's drift has median 0.002-0.007 but spikes to 0.14 -- YOLO-World's
       box wobbling, not the person moving. One spike reset the counter. Fixed
       by asking that most of a rolling window be still.

    2. Counting how many sightings were "still" as a fraction gave 0.58 on real
       falls and lost 7 of 8 -- the wobble is concentrated in a minority of
       frames, so a binary flag throws away the very information that separates
       them. The MEDIAN drift is what the measurements actually showed apart.

    3. Counting DETECTED frames also lost all 8. On the lab clips the detector
       fires on only 91 of 781 inference frames, longest unbroken run 19, so a
       180-frame target was unreachable -- while the person lay there the whole
       time. The detector was blinking, not the person moving.

    So stillness is ELAPSED time, not observations: a track is confirmed when it
    has existed long enough AND the times we did see it were in the same place.
    Drift is divided by the gap since the last sighting, so a blink does not
    read as a jump.
    """

    def __init__(self, min_frames, max_drift, match_iou, max_growth,
                 track_timeout, min_observations=4):
        self.min_frames = max(1, int(min_frames))
        self.max_drift = max_drift
        self.match_iou = match_iou
        self.max_growth = max_growth
        self.track_timeout = track_timeout
        self.min_observations = max(2, int(min_observations))
        self._tracks = []
        self._frame = 0

    @staticmethod
    def _centre_size_area(box):
        x1, y1, x2, y2 = box
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0), math.hypot(w, h), float(w * h)

    def track_info(self, box):
        """Progress of the track matching `box`, for on-screen diagnostics.

        Returns None when nothing matches. Reporting this is the difference
        between a candidate that is silently dropped and one you can watch being
        evaluated -- the whole reason a candidate is drawn at all.
        """
        best, best_iou = None, 0.0
        for t in self._tracks:
            v = iou(box, t["box"])
            if v > best_iou:
                best, best_iou = t, v
        if best is None or best_iou < self.match_iou:
            return None
        w = best["window"]
        return {
            "elapsed": self._frame - best["first"],
            "need": self.min_frames,
            "observations": len(w),
            "min_observations": self.min_observations,
            "median_drift": _median([x[0] for x in w]) if w else None,
            "max_drift": self.max_drift,
            "confirmed": best["confirmed"],
        }

    def update(self, detections):
        """Feed this frame's surviving detections. Returns only those whose
        track has held position long enough."""
        self._frame += 1
        for t in self._tracks:
            t["matched"] = False

        released = []
        for det in detections:
            centre, size, area = self._centre_size_area(det.xyxy)

            best, best_iou = None, 0.0
            for t in self._tracks:
                if t["matched"]:
                    continue
                v = iou(det.xyxy, t["box"])
                if v > best_iou:
                    best, best_iou = t, v

            if best is None or best_iou < self.match_iou:
                self._tracks.append({
                    "box": det.xyxy, "centre": centre, "size": size,
                    "area": area, "window": deque(maxlen=self.min_frames),
                    "confirmed": False, "first": self._frame,
                    "last": self._frame, "matched": True,
                })
                continue

            # per-frame drift: divide by the gap since we last saw it, otherwise
            # a blink in the detector reads as the person jumping
            gap = max(1, self._frame - best["last"])
            drift = math.hypot(centre[0] - best["centre"][0],
                               centre[1] - best["centre"][1]) / \
                max(1.0, (size + best["size"]) / 2.0) / gap
            growth = abs(area - best["area"]) / max(1.0, best["area"]) / gap

            best["window"].append((drift, growth))
            best.update(box=det.xyxy, centre=centre, size=size, area=area,
                        last=self._frame, matched=True)

            elapsed = self._frame - best["first"]
            w = best["window"]
            if elapsed >= self.min_frames and len(w) >= self.min_observations:
                # MEDIAN drift, not a count of "still" frames. A fallen person's
                # box wobbles hard on a minority of frames (p50 0.002-0.007 but
                # p90 up to 0.14); counting flags gave a still-ratio of 0.58 and
                # lost 7 of 8 lab falls, while the median cleanly separates
                # fallen (0.0009-0.0046) from walking (0.0276-0.0373).
                md = _median([x[0] for x in w])
                mg = _median([x[1] for x in w])
                if md <= self.max_drift and mg <= self.max_growth:
                    best["confirmed"] = True
                elif md > self.max_drift * 2 or mg > self.max_growth * 2:
                    best["confirmed"] = False
            if best["confirmed"]:
                released.append(det)

        self._tracks = [t for t in self._tracks
                        if self._frame - t["last"] <= self.track_timeout]
        return released


