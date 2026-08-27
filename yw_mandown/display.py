"""
Screen-fitting and tiling for the live window.

Two problems this solves:

1. Camera frames are often larger than the laptop panel. A 1080p stream shown
   1:1 on a 1080p screen loses its title bar off the bottom; two of them side by
   side is worse. Everything here scales down to fit, never up.

2. cv2.imshow and cv2.waitKey must run on the main thread. On X11, calling them
   from worker threads is a real hang/crash, not a theoretical one. So workers
   hand their latest annotated frame to a LatestFrames slot and the main thread
   does all the drawing -- see cli.run_threaded_with_window.
"""

import math
import os
import shutil
import subprocess
import threading

import cv2
import numpy as np

DEFAULT_SCREEN = (1920, 1080)


def warn_if_qt_fonts_missing():
    """opencv-python ships Qt plugins but not Qt fonts, so QFontDatabase prints
    a wall of warnings at startup. Cosmetic, but it buries real output. Point at
    the fix rather than printing nothing and letting the noise look like a bug."""
    try:
        import cv2
        qt_fonts = os.path.join(os.path.dirname(cv2.__file__), "qt", "fonts")
        qt_dir = os.path.dirname(qt_fonts)
        if os.path.isdir(qt_dir) and not os.path.isdir(qt_fonts):
            print("note: opencv's Qt build has no fonts directory, so you will see "
                  "repeated 'QFontDatabase: Cannot find font directory' warnings.\n"
                  "      Harmless. Silence them with:  python tools/fix_qt_fonts.py")
    except Exception:
        pass


def screen_size():
    """Best-effort desktop resolution. Falls back rather than raising -- a wrong
    guess costs a badly sized window, not a crash."""
    try:
        import tkinter
        root = tkinter.Tk()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        if size[0] > 0 and size[1] > 0:
            return size
    except Exception:
        pass

    if shutil.which("xrandr"):
        try:
            out = subprocess.run(["xrandr"], capture_output=True, text=True,
                                 timeout=5).stdout
            for line in out.splitlines():
                if " connected" in line:
                    for token in line.split():
                        if "x" in token and "+" in token:      # e.g. 1920x1080+0+0
                            wh = token.split("+")[0]
                            w, h = wh.split("x")
                            return int(w), int(h)
        except Exception:
            pass

    return DEFAULT_SCREEN


def fit(frame, max_w, max_h):
    """Scale frame down to fit inside (max_w, max_h), preserving aspect.
    Never scales up -- blowing a 640x480 camera up to fullscreen just makes it
    blurry and costs CPU."""
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0 or max_w <= 0 or max_h <= 0:
        return frame
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def grid_shape(n):
    """Columns x rows for n tiles, biased wide because laptop panels are wide."""
    if n <= 1:
        return 1, 1
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


def tile(frames, labels, max_w, max_h):
    """Lay frames out in a grid that fits inside (max_w, max_h).

    frames may contain None for a source that has not produced a frame yet --
    those become a placeholder tile so the grid does not reflow when a camera
    connects, which would otherwise make the whole window jump.
    """
    n = len(frames)
    if n == 0:
        return None
    cols, rows = grid_shape(n)
    budget_w = max(1, max_w // cols)
    budget_h = max(1, max_h // rows)

    # Size the cells to the scaled content, not to the full budget. Filling the
    # budget would letterbox 3:2 camera frames inside tall cells and waste a
    # third of the window on black bars.
    scaled = [None if f is None else fit(f, budget_w, budget_h) for f in frames]
    real = [x for x in scaled if x is not None]
    if real:
        cell_w = max(x.shape[1] for x in real)
        cell_h = max(x.shape[0] for x in real)
    else:
        cell_w, cell_h = budget_w, budget_h

    cells = []
    for frame, label in zip(scaled, labels):
        if frame is None:
            cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            cv2.putText(cell, f"{label}: waiting...", (12, cell_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2)
        else:
            cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            fh, fw = frame.shape[:2]
            y, x = (cell_h - fh) // 2, (cell_w - fw) // 2
            cell[y:y + fh, x:x + fw] = frame
        cells.append(cell)

    while len(cells) < cols * rows:      # pad the last row
        cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    return np.vstack([np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)])


def window_slots(source_ids, max_w, max_h):
    """One (x, y, w, h) slot per source, laid out so separate windows tile the
    screen instead of stacking on top of each other.

    Window managers place new windows wherever they like -- usually cascaded on
    top of one another, which for a 2-camera live view means one camera hidden
    behind the other. Positioning them explicitly is the whole point.
    """
    n = len(source_ids)
    cols, rows = grid_shape(n)
    cell_w = max(1, max_w // cols)
    cell_h = max(1, max_h // rows)
    slots = {}
    for i, sid in enumerate(source_ids):
        col, row = i % cols, i // cols
        slots[sid] = (col * cell_w, row * cell_h, cell_w, cell_h)
    return slots


def window_budget(cfg_display):
    """How much room the window may use, from config + detected screen."""
    max_w = cfg_display.get("max_width")
    max_h = cfg_display.get("max_height")
    if max_w and max_h:
        return int(max_w), int(max_h)

    if not cfg_display.get("fit_to_screen", True):
        return 10 ** 6, 10 ** 6           # effectively unbounded

    sw, sh = screen_size()
    margin = float(cfg_display.get("screen_margin", 0.9))
    margin = min(max(margin, 0.1), 1.0)
    return int(sw * margin), int(sh * margin)


class LatestFrames:
    """Newest frame per source, overwriting.

    Deliberately drop-oldest rather than a queue: for a live view, showing the
    most recent frame matters and a backlog is worthless. A queue would also let
    a slow display thread pin memory holding frames nobody will ever see.
    """

    def __init__(self, source_ids):
        self._lock = threading.Lock()
        self._frames = {sid: None for sid in source_ids}

    def put(self, source_id, frame):
        with self._lock:
            self._frames[source_id] = frame

    def snapshot(self, source_ids):
        with self._lock:
            return [None if self._frames.get(s) is None else self._frames[s].copy()
                    for s in source_ids]
