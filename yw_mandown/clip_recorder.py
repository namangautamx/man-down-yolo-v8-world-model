"""
Event-triggered clip saving.

Keeps a rolling pre-roll buffer of the last N seconds. When a detection
fires, opens a VideoWriter, flushes the pre-roll into it, then keeps writing
live frames for post_seconds after the *last* detection (repeated detections
extend the clip instead of cutting a new one every frame).
"""

import os
from collections import deque
from datetime import datetime

import cv2


class ClipRecorder:
    def __init__(self, output_dir, source_id, fps, frame_size,
                 pre_seconds, post_seconds):
        self.output_dir = output_dir
        self.source_id = source_id
        self.fps = max(fps, 1.0)
        self.frame_size = frame_size  # (w, h)
        self.pre_frames = max(1, int(self.fps * pre_seconds))
        self.post_frames = max(1, int(self.fps * post_seconds))

        self.buffer = deque(maxlen=self.pre_frames)
        self.writer = None
        self.frames_since_last_trigger = 0
        self.current_path = None

        os.makedirs(output_dir, exist_ok=True)

    def add_frame(self, frame, detected):
        """Call once per processed frame. detected=True on any frame that
        had a surviving positive detection. Returns the clip path if a clip
        was just closed on this call, else None."""
        self.buffer.append(frame)
        closed_path = None

        if detected:
            self.frames_since_last_trigger = 0
            if self.writer is None:
                self._start()

        if self.writer is not None:
            self.writer.write(frame)
            self.frames_since_last_trigger += 1
            if self.frames_since_last_trigger > self.post_frames:
                closed_path = self._close()

        return closed_path

    def _start(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.source_id}_event_{timestamp}.mp4"
        self.current_path = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.current_path, fourcc, self.fps, self.frame_size
        )
        for buffered_frame in self.buffer:
            self.writer.write(buffered_frame)
        print(f"[{self.source_id}] clip started: {self.current_path}")

    def _close(self):
        self.writer.release()
        path = self.current_path
        print(f"[{self.source_id}] clip saved: {path}")
        self.writer = None
        self.current_path = None
        self.frames_since_last_trigger = 0
        return path

    def close(self):
        """Force-close on shutdown so an in-progress clip isn't left corrupt."""
        if self.writer is not None:
            return self._close()
        return None
