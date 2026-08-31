"""
Entry point.

    python -m yw_mandown.cli --config config/config.yaml
    python -m yw_mandown.cli --config config/config.yaml --source cam1

One source enabled -> runs in the main thread, live window allowed (if
configured on).

More than one source enabled -> runs each in its own thread AND shows them
tiled in a single live window, scaled to fit the screen.

The threads never call cv2 display functions. Each worker hands its latest
annotated frame to a LatestFrames slot and the main thread does every
imshow/waitKey -- on X11, calling those from a worker is a real hang/crash,
not a theoretical one. Set display.show_window: false to run headless and just
collect clips, frames and logs/events.jsonl.

The YOLO-World model itself is loaded once and shared across all source
threads. ultralytics/torch calls are serialized behind a lock so concurrent
sources don't stomp on the same forward pass; this trades some throughput
for correctness. It is not a real batcher -- if you outgrow this, batch
frames across sources before calling predict(), the way OWLv2 side of this
project does.
"""

import argparse
import sys
import threading

import cv2

from .config import ConfigError, load_config, enabled_sources
from .detector import SecondPassVeto, YoloWorldDetector
from .display import (LatestFrames, fit, tile, warn_if_qt_fonts_missing,
                      window_budget, window_slots)
from .pipeline import SourcePipeline
from .sources import build_source


class LockedDetector:
    """Wraps YoloWorldDetector so infer() is safe to call from multiple threads."""

    def __init__(self, detector, veto):
        self._detector = detector
        self._veto = veto
        self._lock = threading.Lock()

    def infer(self, frame):
        with self._lock:
            return self._detector.infer(frame)

    def filter_by_geometry(self, positives, min_aspect_ratio, frame_shape=None,
                           edge_mode="none", edge_margin=8):
        return self._detector.filter_by_geometry(
            positives, min_aspect_ratio, frame_shape, edge_mode, edge_margin)

    def dedupe(self, positives, negatives):
        # pure box arithmetic, no model call -- no lock needed
        return self._detector.dedupe(positives, negatives)

    def apply_veto(self, frame, positives):
        with self._lock:
            return self._veto.apply(frame, positives)



def build_detector(cfg):
    return YoloWorldDetector(
        model_name=cfg["model"]["name"],
        positive_prompts=cfg["prompts"]["positive"],
        negative_prompts=cfg["prompts"]["negative"],
        image_size=cfg["model"]["image_size"],
        device=cfg["model"]["device"],
        conf_threshold=cfg["prompts"]["conf_threshold"],
        duplicate_iou=cfg["prompts"]["duplicate_iou"],
        negative_override_iou=cfg["prompts"]["negative_override_iou"],
    )


def build_veto(cfg):
    return SecondPassVeto(
        model_name=cfg["model"]["name"],
        veto_prompts=cfg["prompts"]["veto"],
        image_size=cfg["model"]["image_size"],
        device=cfg["model"]["device"],
        veto_conf=cfg["prompts"]["veto_conf"],
        veto_iou=cfg["prompts"]["veto_iou"],
        veto_containment=cfg["prompts"]["veto_containment"],
    )




def run_single(src_cfg, detector, cfg, veto=None):
    source = build_source(src_cfg)
    pipeline = SourcePipeline(source, detector, cfg, allow_window=True,
                              veto=veto,
                              source_cfg_mount=src_cfg.get("mount", "oblique"))
    return pipeline.run()


def run_threaded_with_window(sources_cfg, detector, cfg, veto=None):
    """Workers capture + infer; this thread owns the window."""
    stop_event = threading.Event()
    ids = [s["id"] for s in sources_cfg]
    latest = LatestFrames(ids)
    results = {}
    threads = []

    def worker(src_cfg):
        source = build_source(src_cfg, stop_flag=stop_event.is_set)
        pipeline = SourcePipeline(source, detector, cfg, allow_window=False,
                                  frame_sink=latest.put, veto=veto,
                                  source_cfg_mount=src_cfg.get("mount", "oblique"))
        try:
            results[src_cfg["id"]] = pipeline.run()
        finally:
            # a camera that dies must not leave the window waiting forever
            results.setdefault(src_cfg["id"], 0)

    for src_cfg in sources_cfg:
        t = threading.Thread(target=worker, args=(src_cfg,), daemon=True,
                             name=src_cfg["id"])
        threads.append(t)
        t.start()

    max_w, max_h = window_budget(cfg["display"])
    title = cfg["display"]["window_name"]
    separate = cfg["display"].get("layout", "separate") == "separate"

    if separate:
        slots = window_slots(ids, max_w, max_h)
        names = {sid: f"{title} | {sid}" for sid in ids}
        for sid in ids:
            x, y, w, h = slots[sid]
            cv2.namedWindow(names[sid], cv2.WINDOW_NORMAL)
            cv2.resizeWindow(names[sid], w, h)
            cv2.moveWindow(names[sid], x, y)
        print(f"Live: {len(ids)} separate windows, each up to "
              f"{slots[ids[0]][2]}x{slots[ids[0]][3]}. 'q' quits, SPACE pauses.")
    else:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        print(f"Live: {len(ids)} sources tiled into at most {max_w}x{max_h}. "
              f"'q' quits, SPACE pauses.")

    paused = False
    try:
        while any(t.is_alive() for t in threads):
            if not paused:
                snap = latest.snapshot(ids)
                if separate:
                    for sid, frame in zip(ids, snap):
                        if frame is not None:
                            x, y, w, h = slots[sid]
                            cv2.imshow(names[sid], fit(frame, w, h))
                else:
                    canvas = tile(snap, ids, max_w, max_h)
                    if canvas is not None:
                        cv2.imshow(title, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                print("stopping all sources...")
                stop_event.set()
                break
            if key == ord(" "):
                paused = not paused
                print("paused" if paused else "resumed")
    except KeyboardInterrupt:
        print("\nCtrl-C received, stopping all sources...")
        stop_event.set()
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=10)
        cv2.destroyAllWindows()

    return results


def run_threaded(sources_cfg, detector, cfg, veto=None):
    stop_event = threading.Event()
    results = {}
    threads = []

    def worker(src_cfg):
        source = build_source(src_cfg, stop_flag=stop_event.is_set)
        pipeline = SourcePipeline(source, detector, cfg, allow_window=False,
                                  veto=veto,
                                  source_cfg_mount=src_cfg.get("mount", "oblique"))
        results[src_cfg["id"]] = pipeline.run()

    for src_cfg in sources_cfg:
        t = threading.Thread(target=worker, args=(src_cfg,), daemon=True, name=src_cfg["id"])
        threads.append(t)
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nCtrl-C received, stopping all sources...")
        stop_event.set()
        for t in threads:
            t.join()

    return results


def main():
    parser = argparse.ArgumentParser(description="YOLO-World man-down detection")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--source", action="append", default=None, metavar="ID",
                         help="run only this source id, ignoring 'enabled'. "
                              "Repeat for several: --source cam1 --source cam2")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1

    sources_cfg = enabled_sources(cfg)
    if args.source:
        wanted = list(dict.fromkeys(args.source))       # de-dup, keep order
        by_id = {s["id"]: s for s in cfg["sources"]}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            print(f"No source with id {missing} in config. "
                  f"Available: {sorted(by_id)}", file=sys.stderr)
            return 1
        sources_cfg = [by_id[w] for w in wanted]

    if not sources_cfg:
        print("No enabled sources. Set enabled: true on at least one source in config.")
        return 1

    if cfg["display"]["show_window"]:
        warn_if_qt_fonts_missing()

    raw_detector = build_detector(cfg)
    veto = build_veto(cfg)

    if len(sources_cfg) == 1:
        run_single(sources_cfg[0], raw_detector, cfg, veto=veto)
    else:
        detector = LockedDetector(raw_detector, veto)
        if cfg["display"]["show_window"]:
            layout = cfg["display"].get("layout", "separate")
            print(f"{len(sources_cfg)} sources -> threaded, "
                  f"{'one window each' if layout == 'separate' else 'tiled in one window'}.")
            run_threaded_with_window(sources_cfg, detector, cfg, veto=veto,
        )
        else:
            print(f"{len(sources_cfg)} sources -> threaded, headless "
                  f"(display.show_window is false).")
            run_threaded(sources_cfg, detector, cfg, veto=veto,
)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
