"""
Append-only audit of EVERY candidate and why it died.

`events.jsonl` records confirmed alarms only. For a safety system the necessary
record is the inverse: when a fall is missed, the question is which stage threw
it away and what number was responsible. Without that, a miss leaves no trace at
all and every tuning decision is guesswork.

One line per candidate per inference frame:

    {"t": "2026-08-29T10:11:12", "source_id": "cam2", "frame": 418,
     "video_time_sec": 27.9, "box": [x1,y1,x2,y2], "conf": 0.51,
     "class": "a person fallen",
     "stage": "geometry",          # which stage decided
     "verdict": "rejected",        # rejected | passed | alarm
     "reason": "aspect",           # machine-readable cause
     "measured": 0.64, "limit": 0.7}

Stages, in pipeline order:
    detector    a box the model emitted (verdict=passed, or rejected/negative_class)
    conf        below prompts.conf_threshold
    geometry    aspect ratio or frame-edge clipping
    veto        an animal explained the box better
    flicker     not enough repeat evidence yet
    stillness   still accumulating, or moving too much
    alarm       survived everything

Volume: one line per candidate per inference frame, so a busy camera writes a
few lines a second. Rotation is the operator's job (logrotate); this module only
appends, and never blocks the pipeline -- a logging failure must not stop
detection, so writes are wrapped and failures counted rather than raised.
"""

import json
import os
from datetime import datetime


class AuditLogger:
    def __init__(self, path, enabled=True):
        self.path = path
        self.enabled = enabled and bool(path)
        self.write_errors = 0
        if self.enabled:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, source_id, frame_number, video_time_sec, detection,
            stage, verdict, reason=None, measured=None, limit=None):
        if not self.enabled:
            return
        rec = {
            "t": datetime.now().isoformat(timespec="seconds"),
            "source_id": source_id,
            "frame": frame_number,
            "video_time_sec": round(video_time_sec, 2),
            "stage": stage,
            "verdict": verdict,
        }
        if detection is not None:
            rec["box"] = list(detection.xyxy)
            rec["conf"] = round(float(detection.conf), 4)
            rec["class"] = detection.class_name
        if reason is not None:
            rec["reason"] = reason
        if measured is not None:
            rec["measured"] = round(float(measured), 4)
        if limit is not None:
            rec["limit"] = round(float(limit), 4)
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            # never let the audit trail take the detector down with it
            self.write_errors += 1
