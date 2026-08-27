"""
Measure how inference latency scales with the number of prompts, on whatever
machine you run it on.

    python tools/bench_prompts.py
    python tools/bench_prompts.py --video /path/to/clip.mp4 --imgsz 640

Written because "does adding prompts slow it down?" has a different answer on
every device, and the honest way to answer it for a Jetson is to run it there.

Reference numbers, RTX 5050 Laptop, imgsz=640, yolov8m-worldv2:

     3 classes  14.52 ms  68.9 fps   baseline
     7 classes  14.24 ms  70.2 fps   -1.9%
    23 classes  14.50 ms  68.9 fps   -0.1%
    50 classes  14.78 ms  67.7 fps   +1.8%

i.e. flat. The CLIP text encoder runs once inside set_classes() at startup, not
per frame; per frame the backbone dominates and the class comparison is a dot
product. On a slower GPU the backbone takes even longer, so the relative cost of
extra classes should shrink, not grow -- but measure, don't assume.

Jetson notes:
  * Run `sudo nvpmodel -q` and `sudo jetson_clocks` first. Power mode changes
    inference time far more than prompt count does; benchmarking in a throttled
    mode measures the throttle, not the model.
  * Watch `tegrastats` in another shell for thermal throttling over a long run.
  * If you later export to TensorRT/ONNX, the class count is frozen into the
    engine at export time -- changing prompts then means re-exporting, and the
    per-frame cost question changes shape entirely. This script measures the
    PyTorch path.
"""

import argparse
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])
from yw_mandown.detector import YoloWorldDetector  # noqa: E402

POSITIVES = [
    "a person lying on the floor",
    "a person lying on the ground",
    "a person sleeping",
]

# generic filler, only there to pad the class count
FILLER = [
    "a person standing", "a person walking", "a mattress", "a pile of bags",
    "a chair", "a table", "a dog", "a car", "a bicycle", "a backpack",
    "a bottle", "a laptop", "a sofa", "a bed", "a plant", "a door", "a window",
    "a lamp", "a book", "a phone", "a cup", "a shoe", "a hat", "a bag", "a box",
    "a ladder", "a bucket", "a broom", "a fan", "a clock", "a mirror", "a rug",
    "a curtain", "a shelf", "a basket", "a towel", "a pillow", "a blanket",
    "a jacket", "a helmet", "a glove", "a rope", "a pipe", "a barrel",
    "a crate", "a trolley", "a pallet",
]


def describe_device():
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu", "CPU only (no CUDA)"
        name = torch.cuda.get_device_name(0)
        try:
            with open("/proc/device-tree/model") as f:
                name += f"  [{f.read().strip(chr(0))}]"
        except OSError:
            pass
        return "cuda", name
    except ImportError:
        return "cpu", "torch not importable"


def load_frames(video, n, imgsz):
    if not video:
        rng = np.random.default_rng(0)
        return [rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(n)]
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * i / n))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        sys.exit(f"could not read any frames from {video}")
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None, help="real footage; omit for synthetic frames")
    ap.add_argument("--model", default="yolov8m-worldv2.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--counts", default="3,4,5,7,10,15,23,33,50",
                    help="comma-separated TOTAL class counts to test")
    args = ap.parse_args()

    device, desc = describe_device()
    frames = load_frames(args.video, args.frames, args.imgsz)
    counts = sorted({max(len(POSITIVES), int(c)) for c in args.counts.split(",")})

    print(f"device : {desc}")
    print(f"model  : {args.model}   imgsz={args.imgsz}")
    print(f"frames : {len(frames)} x {args.repeats} repeats"
          f"{'' if args.video else '  (SYNTHETIC -- use --video for realistic box counts)'}")
    print()
    print(f"{'classes':>8} {'median ms':>10} {'mean ms':>9} {'p95 ms':>8} {'fps':>7}   vs {counts[0]}")

    sync = (lambda: None)
    if device == "cuda":
        import torch
        sync = torch.cuda.synchronize

    base = None
    for total in counts:
        negs = FILLER[:total - len(POSITIVES)]
        det = YoloWorldDetector(args.model, POSITIVES, negs,
                                args.imgsz, None, 0.25)
        for f in frames[:10]:
            det.infer(f)                       # warmup
        sync()

        times = []
        for _ in range(args.repeats):
            for f in frames:
                t = time.perf_counter()
                det.infer(f)
                sync()
                times.append((time.perf_counter() - t) * 1000)

        med = statistics.median(times)
        p95 = sorted(times)[int(len(times) * 0.95)]
        if base is None:
            base = med
        delta = "baseline" if med == base else f"{(med / base - 1) * 100:+.1f}%"
        print(f"{total:>8} {med:>10.2f} {statistics.mean(times):>9.2f} "
              f"{p95:>8.2f} {1000 / med:>7.1f}   {delta}")

    print("\nIf the rightmost column stays within a few percent, prompt count is "
          "free on this device and you should tune prompts for accuracy alone.")


if __name__ == "__main__":
    main()
