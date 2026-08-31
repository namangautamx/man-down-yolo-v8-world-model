"""
Two gates between the model and an alarm, both strictly subtractive.

TemporalConfirmer  requires repeated evidence before an alarm, so a
                   one-frame flicker cannot raise one.
StillnessGate      requires the box to hold position, so a moving person
                   cannot.

Neither may ever produce a box on a frame the model did not fire on. An
earlier TemporalConfirmer had a "hold" that kept an alarm alive across empty
frames and redrew the last box; that invented detections and was removed.
"""
import math

from collections import deque

from .detector import iou


class TemporalConfirmer:
    """Require N sightings of THE SAME BOX within the last M inference frames.

    Purpose is narrow: stop a single-frame flicker from raising an alarm. On the
    cam1 live run, 158 detected frames were spread over 29 separate runs, 13 of
    them one frame long -- those are what this removes.

    PER BOX, not per frame
    ----------------------
    This gate used to hold one boolean window for the whole camera: "did ANY
    detection appear in this frame". Once any box satisfied N-of-M the gate
    opened, and it then passed every OTHER box in the frame straight through --
    including one appearing for the very first time. Demonstrated:

        frame 1  [A]      A is a genuine fall
        frame 2  [A]      2-of-4 satisfied, gate opens
        frame 3  [A, B]   B has never been seen before
                 -> both released; B was never filtered at all

    On a busy camera one real fall therefore disabled the flicker filter for
    everything else in view, permanently. The audit log shows what that cost:
    152 flicker rejections against 729 alarms, so it was barely filtering.

    Each box now carries its own N-of-M window, matched frame to frame by IoU,
    exactly like StillnessGate. A box must earn its own repeat evidence and can
    inherit nothing from another.

    STRICTLY SUBTRACTIVE. update() returns a subset of the detections passed in.
    It never returns a stored box, never extends an alarm across a frame the
    model was silent on, and never invents anything. An older version had a
    "hold" that did exactly that; it is gone. Keep it gone.

    Deliberately short: the stillness gate already imposes the long wait, so a
    second long confirmation on top would only delay a real alarm. This is a
    flicker filter, not a second opinion.
    """

    def __init__(self, confirm_frames, window_frames, match_iou=0.30):
        if window_frames < confirm_frames:
            raise ValueError("window_frames must be >= confirm_frames")
        self.confirm_frames = confirm_frames
        self.window_frames = window_frames
        self.match_iou = match_iou
        self._tracks = []

    def reset(self):
        """Forget everything. Called at a video-folder cut, where the next frame
        is an unrelated scene and no track may survive."""
        self._tracks = []

    def hits(self, box):
        """How many of the last window_frames this box was seen in, for the
        audit log -- so a rejection records the number responsible."""
        i = self._match(box, used=())
        return sum(self._tracks[i]["window"]) if i is not None else 0

    def _match(self, box, used):
        """Index of the best-overlapping unused track, or None. Does not claim
        it -- the caller decides whether to."""
        best, best_iou = None, 0.0
        for i, t in enumerate(self._tracks):
            if i in used:
                continue
            v = iou(box, t["box"])
            if v > best_iou:
                best, best_iou = i, v
        return best if best_iou >= self.match_iou else None

    def update(self, detections):
        """Feed one inference frame's surviving detections. Returns the ones
        whose OWN window now holds enough sightings."""
        # every existing track is assumed unseen until a detection claims it,
        # so a box that stops appearing ages out of its own window
        for t in self._tracks:
            t["window"].append(False)

        used = set()
        released = []
        for det in detections:
            i = self._match(det.xyxy, used)
            if i is None:
                t = {"box": det.xyxy,
                     "window": deque([True], maxlen=self.window_frames)}
                self._tracks.append(t)
            else:
                used.add(i)
                t = self._tracks[i]
                t["window"][-1] = True
                t["box"] = det.xyxy
            if sum(t["window"]) >= self.confirm_frames:
                released.append(det)

        # a track with no sighting anywhere in its window is dead
        self._tracks = [t for t in self._tracks if any(t["window"])]
        return released


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

    Those were per INFERENCE FRAME. Drift is now measured per SECOND of
    video, so the limit means the same thing on a 15fps camera and a 120fps
    one. The configured value is scaled accordingly.

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

    def __init__(self, still_seconds, max_drift, match_iou, max_growth,
                 track_timeout_seconds, min_observations=4, window_cap=240):
        self.still_seconds = max(0.0, float(still_seconds))
        self.max_drift = max_drift
        self.match_iou = match_iou
        self.max_growth = max_growth
        self.track_timeout_seconds = max(0.0, float(track_timeout_seconds))
        self.min_observations = max(2, int(min_observations))
        self.window_cap = max(4, int(window_cap))
        self._tracks = []
        self._now = 0.0

    def reset(self):
        """Forget every track. Called at a video-folder cut, where the next
        frame is an unrelated scene and no track may survive."""
        self._tracks = []

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
            "elapsed": round(self._now - best["first"], 2),
            "need": self.still_seconds,
            "observations": len(w),
            "min_observations": self.min_observations,
            "median_drift": _median([x[0] for x in w]) if w else None,
            "max_drift": self.max_drift,
            "confirmed": best["confirmed"],
        }

    def update(self, detections, now_sec=None):
        """Feed this frame's surviving detections and the video timestamp.

        Time comes from the VIDEO, not from a frame count. Counting frames made
        every threshold depend on the source frame rate and on how fast the GPU
        happened to be running: still_seconds 6.0 meant 360 inference frames on
        120fps footage (impossible in an 781-frame clip) and 45 on a 15fps
        camera. Same setting, two completely different gates. Timestamps make
        the setting mean one thing everywhere.
        """
        self._now = float(now_sec) if now_sec is not None else self._now + 1.0
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
                    "area": area, "window": deque(maxlen=self.window_cap),
                    "confirmed": False, "first": self._now,
                    "last": self._now, "matched": True, "anchor": None,
                })
                continue

            # per-frame drift: divide by the gap since we last saw it, otherwise
            # a blink in the detector reads as the person jumping
            gap = max(1e-3, self._now - best["last"])   # seconds since last seen
            drift = math.hypot(centre[0] - best["centre"][0],
                               centre[1] - best["centre"][1]) / \
                max(1.0, (size + best["size"]) / 2.0) / gap
            growth = abs(area - best["area"]) / max(1.0, best["area"]) / gap

            best["window"].append((drift, growth))
            best.update(box=det.xyxy, centre=centre, size=size, area=area,
                        last=self._now, matched=True)

            elapsed = self._now - best["first"]
            w = best["window"]
            if elapsed >= self.still_seconds and len(w) >= self.min_observations:
                # MEDIAN drift, not a count of "still" frames. A fallen person's
                # box wobbles hard on a minority of frames (p50 0.002-0.007 but
                # p90 up to 0.14); counting flags gave a still-ratio of 0.58 and
                # lost 7 of 8 lab falls, while the median cleanly separates
                # fallen (0.0009-0.0046) from walking (0.0276-0.0373).
                md = _median([x[0] for x in w])
                mg = _median([x[1] for x in w])
                if md <= self.max_drift and mg <= self.max_growth:
                    if not best["confirmed"]:
                        # remember WHERE it earned this. See the anchor check.
                        best["anchor"] = det.xyxy
                    best["confirmed"] = True
                elif md > self.max_drift * 2 or mg > self.max_growth * 2:
                    best["confirmed"] = False
                    best["anchor"] = None

            # A confirmed track may only keep its status while it is still the
            # thing that earned it. `confirmed` is otherwise permanent: the only
            # release path above needs median drift over 2x max_drift, and the
            # median is taken over up to window_cap sightings, so it moves far
            # too slowly to ever fire. Two real consequences:
            #
            #   * the fallen person gets up and walks off, but the track keeps
            #     matching them (IoU only has to reach match_iou) and keeps
            #     alarming while they walk
            #   * someone else steps into that spot and is released INSTANTLY,
            #     never having waited still_seconds themselves
            #
            # Anchoring to the box at confirmation time closes both. A genuinely
            # stationary casualty keeps near-perfect overlap with where it was
            # confirmed, so this cannot cost a real detection; anything that has
            # moved off that spot must earn confirmation again from scratch.
            if best["confirmed"] and best["anchor"] is not None:
                if iou(det.xyxy, best["anchor"]) < self.match_iou:
                    best["confirmed"] = False
                    best["anchor"] = None
                    best["first"] = self._now      # re-earn the wait in the new
                    best["window"].clear()         # position, from zero

            if best["confirmed"]:
                released.append(det)

        self._tracks = [t for t in self._tracks
                        if self._now - t["last"] <= self.track_timeout_seconds]
        return released


