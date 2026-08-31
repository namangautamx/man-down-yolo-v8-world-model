"""
Event-triggered clip saving, encoded on a worker thread.

Keeps a rolling pre-roll buffer of the last N seconds. When a detection fires,
opens a VideoWriter, flushes the pre-roll into it, then keeps writing live
frames for post_seconds after the *last* detection (repeated detections extend
the clip instead of cutting a new one every frame).

Why the worker thread
---------------------
This used to encode on the caller's thread -- the same thread that reads frames
and runs the detector. Measured at 2560x1440:

    clip start (flushing 45 buffered frames)   536 ms   ONE-OFF, at alarm start
    per frame while recording                       11 ms   EVERY frame

Against a 16 ms inference budget that meant the pipeline stalled for half a
second the instant an alarm fired, then ran at under half speed for the whole
duration of the clip. On RTSP the stream backs up behind that stall. It is
exactly the "it lags after detecting the man down" symptom.

Encoding now happens on a worker thread fed by a bounded queue. add_frame()
does a buffer append and a queue put, and returns immediately.

The queue is BOUNDED and drops on overflow. A disk that cannot keep up must
degrade the recording, never the detector -- dropping clip frames loses
evidence, blocking the pipeline loses detections. Drops are counted and
reported so silent loss is visible.
"""

import os
import queue
import threading
from collections import deque
from datetime import datetime

import cv2


class ClipRecorder:
    def __init__(self, output_dir, source_id, fps, frame_size,
                 pre_seconds, post_seconds, max_pending=120):
        self.output_dir = output_dir
        self.source_id = source_id
        self.fps = max(fps, 1.0)
        self.frame_size = frame_size  # (w, h)
        self.pre_frames = max(1, int(self.fps * pre_seconds))
        self.post_frames = max(1, int(self.fps * post_seconds))

        self.buffer = deque(maxlen=self.pre_frames)
        self.frames_since_last_trigger = 0
        self.current_path = None
        self.recording = False
        self.dropped = 0

        os.makedirs(output_dir, exist_ok=True)

        # ("open", path) | ("frame", img) | ("close", None) | ("stop", None)
        self._q = queue.Queue(maxsize=max_pending)
        self._writer = None                 # touched only by the worker
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name=f"clipwriter-{source_id}")
        self._worker.start()

    # ---------------- caller side: never blocks on encoding ----------------

    def add_frame(self, frame, detected):
        """Call once per processed frame. Returns the clip path if a clip was
        just closed on this call, else None."""
        self.buffer.append(frame)
        closed_path = None

        if detected:
            self.frames_since_last_trigger = 0
            if not self.recording:
                self._begin()

        if self.recording:
            self._send(("frame", frame))
            self.frames_since_last_trigger += 1
            if self.frames_since_last_trigger > self.post_frames:
                closed_path = self.current_path
                self._send(("close", None))
                self.recording = False
                self.current_path = None
                self.frames_since_last_trigger = 0

        return closed_path

    def _begin(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.current_path = os.path.join(
            self.output_dir, f"{self.source_id}_event_{timestamp}.mp4")
        self.recording = True
        self._send(("open", self.current_path))
        # hand the pre-roll over as frames; the worker encodes them, not us
        for buffered in list(self.buffer):
            self._send(("frame", buffered))
        print(f"[{self.source_id}] clip started: {self.current_path}")

    def _send(self, item):
        """Queue work, dropping rather than blocking the detector."""
        try:
            self._q.put_nowait(item)
        except queue.Full:
            # only frames are droppable; control messages must get through
            if item[0] == "frame":
                self.dropped += 1
                if self.dropped % 100 == 1:
                    print(f"[{self.source_id}] clip writer behind, dropped "
                          f"{self.dropped} frames from the recording "
                          f"(detection is unaffected)")
            else:
                self._q.put(item)          # block only for open/close/stop

    # ---------------- worker side: all encoding happens here ----------------

    def _run(self):
        while True:
            kind, payload = self._q.get()
            try:
                if kind == "open":
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self._writer = cv2.VideoWriter(payload, fourcc, self.fps,
                                                   self.frame_size)
                elif kind == "frame":
                    if self._writer is not None:
                        self._writer.write(payload)
                elif kind == "close":
                    self._finish()
                elif kind == "stop":
                    self._finish()
                    return
            except Exception as e:                      # never kill the thread
                print(f"[{self.source_id}] clip writer error: {e}")
            finally:
                self._q.task_done()

    def _finish(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def close(self):
        """Force-close on shutdown so an in-progress clip is not left corrupt."""
        path = self.current_path
        self._send(("stop", None))
        self._worker.join(timeout=15.0)
        self.recording = False
        self.current_path = None
        if self.dropped:
            print(f"[{self.source_id}] {self.dropped} frames were dropped from "
                  f"recordings while the writer was behind")
        return path
