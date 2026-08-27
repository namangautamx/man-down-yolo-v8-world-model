"""Append-only JSONL event log, one line per detection."""

import json
import os
from datetime import datetime


class EventLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, source_id, frame_number, video_time_sec, detection, clip_path=None):
        record = {
            "wall_time": datetime.now().isoformat(timespec="seconds"),
            "source_id": source_id,
            "frame_number": frame_number,
            "video_time_sec": round(video_time_sec, 2),
            "class": detection.class_name,
            "conf": round(float(detection.conf), 4),
            "box": list(detection.xyxy),
            "clip_path": clip_path,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
