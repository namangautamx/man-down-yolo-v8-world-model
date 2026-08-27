"""
Frame sources. Each yields (frame_number, timestamp_sec, frame) tuples.

Frame skipping for inference cost is a pipeline decision, not a source one --
sources always hand over every frame, so display and clip recording stay
smooth even when inference only runs on every Nth frame.
"""

import time
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


class VideoFileSource:
    """One video file, played once."""

    def __init__(self, path, source_id):
        self.path = str(path)
        self.source_id = source_id
        self.name = Path(path).name

    def __iter__(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            print(f"[{self.source_id}] could not open {self.path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        try:
            frame_number = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame_number, frame_number / fps, frame
                frame_number += 1
        finally:
            cap.release()

    def fps_hint(self):
        cap = cv2.VideoCapture(self.path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return fps


class VideoFolderSource:
    """Every video file in a folder, played back to back, one source id."""

    def __init__(self, folder, source_id):
        self.folder = folder
        self.source_id = source_id
        self.files = sorted(
            p for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        self._fps = None
        if not self.files:
            print(f"[{source_id}] no video files found in {folder}")

    def __iter__(self):
        global_frame = 0
        fps = self.fps_hint()  # hoisted: this opens a VideoCapture, so calling it
                               # per frame cost ~0.9ms x every frame in the folder
        for path in self.files:
            print(f"[{self.source_id}] -> {path.name}")
            for _, _, frame in VideoFileSource(path, self.source_id):
                yield global_frame, global_frame / fps, frame
                global_frame += 1

    def fps_hint(self):
        if not self.files:
            return 25.0
        if self._fps is None:
            cap = cv2.VideoCapture(str(self.files[0]))
            self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
        return self._fps


class RTSPSource:
    """Live RTSP stream. Reconnects on frame loss rather than dying.

    Runs until stop_flag() returns True (checked between frames), so the
    pipeline can shut it down cleanly (e.g. on 'q' or Ctrl-C) instead of
    the source living forever.
    """

    def __init__(self, url, source_id, reconnect_delay=2.0, stop_flag=None):
        self.url = url
        self.source_id = source_id
        self.reconnect_delay = reconnect_delay
        self.stop_flag = stop_flag or (lambda: False)

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def __iter__(self):
        cap = self._open()
        if not cap.isOpened():
            print(f"[{self.source_id}] cannot open RTSP stream: {self.url}")
            return

        frame_number = 0
        start = time.time()
        try:
            while not self.stop_flag():
                ret, frame = cap.read()
                if not ret:
                    print(f"[{self.source_id}] frame lost, reconnecting...")
                    cap.release()
                    time.sleep(self.reconnect_delay)
                    cap = self._open()
                    continue
                yield frame_number, time.time() - start, frame
                frame_number += 1
        finally:
            cap.release()

    def fps_hint(self):
        # RTSP fps is often unreliable/unreported; assume a sane default,
        # used only for pre-roll buffer sizing on clips.
        return 15.0


def build_source(src_cfg, stop_flag=None):
    stype = src_cfg["type"]
    sid = src_cfg["id"]
    if stype == "video_file":
        return VideoFileSource(src_cfg["path"], sid)
    if stype == "video_folder":
        return VideoFolderSource(src_cfg["path"], sid)
    if stype == "rtsp":
        return RTSPSource(
            src_cfg["url"], sid,
            reconnect_delay=src_cfg.get("reconnect_delay_seconds", 2.0),
            stop_flag=stop_flag,
        )
    raise ValueError(f"unknown source type: {stype}")
