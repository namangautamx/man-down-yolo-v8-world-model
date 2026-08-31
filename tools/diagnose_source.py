"""
Show WHY a source is not alarming -- including the detections that were thrown
away, which the normal pipeline discards silently.

    python tools/diagnose_source.py --source cam2 --seconds 60
    python tools/diagnose_source.py --source cam2 --seconds 60 --save-all

Run it, then get into the pose that is not being detected. Every raw detection
is reported with the stage that killed it and the number responsible, so
"it didn't detect me" turns into "w/h was 0.74, the gate wants 1.0".

Stages, in pipeline order:
    conf      below prompts.conf_threshold          (never leaves the model)
    aspect    geometry.min_aspect_ratio
    edge      geometry.reject_edge_touching
    veto      prompts.veto (dog/cat second pass)
    temporal  temporal.confirm_frames of window_frames
"""

import argparse
import collections
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yw_mandown.config import ConfigError, load_config          # noqa: E402
from yw_mandown.detector import (SecondPassVeto, YoloWorldDetector,  # noqa: E402
                                 containment, iou)
from yw_mandown.sources import build_source                     # noqa: E402
from yw_mandown.temporal import TemporalConfirmer               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--source", required=True, help="source id from the config")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--scan-conf", type=float, default=0.05,
                    help="scan this low so sub-threshold detections are visible too")
    ap.add_argument("--out", default="output/diagnose")
    ap.add_argument("--save-all", action="store_true",
                    help="also save frames where nothing was detected at all")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        sys.exit(str(e))

    src_cfg = next((s for s in cfg["sources"] if s["id"] == args.source), None)
    if src_cfg is None:
        sys.exit(f"no source '{args.source}'. have: "
                 f"{sorted(s['id'] for s in cfg['sources'])}")

    P, G, T = cfg["prompts"], cfg["geometry"], cfg["temporal"]
    out_dir = os.path.join(args.out, args.source)
    os.makedirs(out_dir, exist_ok=True)

    # scan below the real threshold so we can SEE what the threshold rejects
    scan = min(args.scan_conf, P["conf_threshold"])
    det = YoloWorldDetector(cfg["model"]["name"], P["positive"], P["negative"],
                            cfg["model"]["image_size"], cfg["model"]["device"], scan)
    veto = SecondPassVeto(cfg["model"]["name"], P["veto"], cfg["model"]["image_size"],
                          cfg["model"]["device"], P["veto_conf"], P["veto_iou"],
                          P["veto_containment"])
    # hold_frames was removed with the "hold" mechanism; reading it here made
    # this tool crash on startup, i.e. exactly when you needed it.
    conf_t = TemporalConfirmer(T["confirm_frames"], T["window_frames"],
                               T["match_iou"])

    print(f"\nsource {args.source}  imgsz={cfg['model']['image_size']}  "
          f"conf_threshold={P['conf_threshold']}  min_aspect_ratio={G['min_aspect_ratio']}"
          f"  edge={G['reject_edge_touching']}")
    print(f"scanning down to {scan} so rejected detections are visible")
    print(f"running {args.seconds:.0f}s -- get into the pose now. "
          f"frames -> {out_dir}\n")

    source = build_source(src_cfg, stop_flag=lambda: False)
    reasons = collections.Counter()
    saved = 0
    alarms = 0
    t0 = time.time()
    skip = max(1, cfg["processing"]["frame_skip"])

    for frame_number, _, frame in source:
        if time.time() - t0 > args.seconds:
            break
        if frame_number % skip:
            continue

        raw, negatives, shadows = det.infer(frame)
        h, w = frame.shape[:2]
        verdicts = []
        survivors = []

        # a distractor class claimed a region with no positive box near it --
        # a real casualty can be erased here with no other trace
        for sh in shadows:
            verdicts.append((sh, "negative_class_shadow",
                             f"claimed by {sh.class_name!r} at {sh.conf:.2f}"))



        for p in raw:
            x1, y1, x2, y2 = p.xyxy
            ar = (x2 - x1) / max(1, y2 - y1)
            if p.conf < P["conf_threshold"]:
                verdicts.append((p, "conf", f"{p.conf:.2f} < {P['conf_threshold']}"))
                continue
            if G["min_aspect_ratio"] and ar < G["min_aspect_ratio"]:
                verdicts.append((p, "aspect", f"w/h {ar:.2f} < {G['min_aspect_ratio']}"))
                continue
            if YoloWorldDetector._clipped_by_edge(p.xyxy, frame.shape,
                                                  G["reject_edge_touching"],
                                                  G["edge_margin_px"]):
                verdicts.append((p, "edge", f"box touches frame border ({w}x{h})"))
                continue
            survivors.append(p)

        # same order as the pipeline: dedupe AFTER geometry
        survivors, suppressed = det.dedupe(survivors, negatives)
        for d, why, measured in suppressed:
            verdicts.append((d, why.split(":")[0], f"{why} ({measured:.2f})"))

        if survivors and veto.enabled:
            kept, vetoes = veto.apply(frame, survivors)
            keptids = {id(k) for k in kept}
            for p in survivors:
                if id(p) not in keptids:
                    best = max(vetoes, key=lambda v: max(iou(p.xyxy, v.xyxy),
                                                         containment(v.xyxy, p.xyxy)))
                    verdicts.append((p, "veto", f"{best.class_name} {best.conf:.2f}"))
            survivors = kept

        # update() returns the surviving subset -- it used to be unpacked as a
        # 2-tuple, which raised on every frame.
        confirmed = conf_t.update(survivors)
        on = bool(confirmed)
        if on:
            alarms += 1
        held = {id(c) for c in confirmed}
        for p in survivors:
            if id(p) not in held:
                verdicts.append((p, "temporal",
                                 f"seen {conf_t.hits(p.xyxy)}x, needs "
                                 f"{T['confirm_frames']} of last "
                                 f"{T['window_frames']} frames"))

        for _, why, _ in verdicts:
            reasons[why] += 1
        if on:
            reasons["ALARM"] += 1

        if (verdicts or args.save_all) and saved < 40:
            img = frame.copy()
            for p, why, detail in verdicts:
                x1, y1, x2, y2 = p.xyxy
                col = (0, 165, 255) if why != "ALARM" else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                cv2.putText(img, f"{why}: {detail}", (x1, max(18, y1 - 8)),
                            0, 0.55, col, 2)
            for p in survivors:
                x1, y1, x2, y2 = p.xyxy
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, f"PASSED {p.conf:.2f}", (x1, max(18, y1 - 8)),
                            0, 0.6, (0, 255, 0), 2)
            tag = "alarm" if on else (verdicts[0][1] if verdicts else "none")
            cv2.imwrite(f"{out_dir}/{saved:03d}_{tag}_f{frame_number}.jpg", img)
            saved += 1

    print("what happened to each detection:")
    for why, n in reasons.most_common():
        label = {"conf": "dropped: confidence too low",
                 "aspect": "dropped: box not wide enough (min_aspect_ratio)",
                 "edge": "dropped: box clipped by frame border",
                 "veto": "dropped: animal veto",
                 "temporal": "held back: not enough repeat sightings of that box",
                 "duplicate_box": "merged: same object already had a box",
                 "negative_override": "dropped: a distractor scored higher on the same box",
                 "ALARM": "ALARM RAISED"}.get(why, why)
        print(f"   {label:<48} {n}")
    if not reasons:
        print("   nothing detected at all, even at the low scan threshold")
    print(f"\n{saved} annotated frames in {out_dir}")
    print("orange = rejected (with the reason), green = passed")


if __name__ == "__main__":
    main()
