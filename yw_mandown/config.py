"""
Config loading + validation.

Everything the pipeline does is driven from config/config.yaml. This module's
job is to load it, fill in defaults, and fail loudly (before any model is
loaded, before any camera is opened) if something required is missing or the
wrong type -- rather than crashing three minutes into a run.
"""

import os
import re

import yaml


class ConfigError(Exception):
    pass


DEFAULTS = {
    "model": {
        "name": "yolov8m-worldv2.pt",
        "device": None,          # None -> let ultralytics pick (cuda if available)
        "image_size": 640,
    },
    "prompts": {
        "positive": [],
        "negative": [],
        "conf_threshold": 0.70,
        "veto": [],              # second-pass classes that can veto a detection
        "veto_conf": 0.45,
        "veto_iou": 0.30,
        "veto_containment": 0.60,
        # ultralytics runs NMS per class, so one person yields one box per
        # prompt that fires on them. Merge boxes overlapping this much.
        "duplicate_iou": 0.60,
        # drop a positive when a distractor claims essentially the same box AND
        # outscores it. DEFAULT OFF -- measured on the 24:58 casualty in
        # eqiom_video, the model preferred "a person standing" over "a person
        # fallen" on the same box every frame he was on the ground. See
        # YoloWorldDetector._dedupe.
        "negative_override_iou": 0.0,
    },
    "geometry": {
        "min_aspect_ratio": 1.0,        # 0 / null disables the check
        "reject_edge_touching": "bottom",  # "none" | "bottom" | "any"
        "edge_margin_px": 8,
    },
    "temporal": {
        "enabled": True,
        "confirm_frames": 2,     # need this many detections...
        "window_frames": 4,      # ...within this many recent inference frames
        "match_iou": 0.30,       # ...OF THE SAME BOX, matched frame to frame
    },
    "stillness": {
        "enabled": True,
        "still_seconds": 3.0,    # how long a box must hold position before it alarms
        "max_drift": 0.02,       # centroid movement per inference frame, in box widths
        "max_growth": 0.03,      # box area change per inference frame
        "match_iou": 0.30,       # IoU needed to call it the same box next frame
        "track_timeout_seconds": 15.0,
        "min_observations": 4,   # never confirm on fewer sightings than this
    },
    "processing": {
        "frame_skip": 2,
    },
    "output": {
        "base_folder": "./output",
        "save_frames": True,
        "save_clips": True,
        "clip_pre_seconds": 3.0,
        "clip_post_seconds": 3.0,
        "frame_min_interval_seconds": 1.0,
    },
    "display": {
        "show_window": True,
        "window_name": "MAN-DOWN DETECTION",
        "show_candidates": True,  # draw unconfirmed boxes yellow, with their numbers
        "layout": "separate",    # "separate" = one window per camera, "grid" = all in one
        "fit_to_screen": True,   # scale the window down to the detected screen
        "screen_margin": 0.9,    # use at most this fraction of it
        "max_width": None,       # set both to pin an explicit size instead
        "max_height": None,
    },
    "logging": {
        "events_path": "./logs/events.jsonl",
        "audit_path": "./logs/audit.jsonl",   # every candidate + why it died
        "audit_enabled": True,
        "stale_after_seconds": 30.0,   # warn if a source produces no frame for this long
        "measured_inf_per_sec": None,  # set from tools/bench_prompts.py on the deployment GPU
    },
    "sources": [],
}


def _merge_defaults(user_cfg, defaults):
    """Recursively fill in missing keys from defaults. Does not touch lists."""
    if not isinstance(user_cfg, dict):
        return defaults
    merged = dict(defaults)
    for key, default_val in defaults.items():
        if key in user_cfg:
            if isinstance(default_val, dict) and isinstance(user_cfg[key], dict):
                merged[key] = _merge_defaults(user_cfg[key], default_val)
            else:
                merged[key] = user_cfg[key]
        else:
            merged[key] = default_val
    # keep any extra user keys we don't know about, rather than silently dropping them
    for key, val in user_cfg.items():
        if key not in defaults:
            merged[key] = val
    return merged


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value, where, errors):
    """Replace ${VAR} with the environment value.

    Credentials must not live in the config file: config/config.yaml is tracked
    by git, so anything written here is committed. Unset variables are a config
    ERROR rather than an empty substitution -- a camera silently authenticating
    with a blank password would fail at connect time, far from the cause.
    """
    if not isinstance(value, str):
        return value

    def sub(m):
        name = m.group(1)
        got = os.environ.get(name)
        if got is None:
            errors.append(f"{where}: environment variable ${{{name}}} is not set")
            return m.group(0)
        return got

    return _ENV_PATTERN.sub(sub, value)


def load_config(path):
    if not os.path.exists(path):
        raise ConfigError(
            f"Config file not found: {path}\n"
            f"Copy config/config.example.yaml to config/config.yaml and edit it."
        )

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    _migrate_legacy_keys(raw)
    _warn_removed_keys(raw)
    cfg = _merge_defaults(raw, DEFAULTS)

    _validate(cfg, path)
    return cfg


# save_crops/crop_min_interval_seconds were renamed when crop saving became
# whole-frame saving. Accept the old spellings so an existing config keeps
# working instead of silently falling back to defaults.
_RENAMED = {
    "save_crops": "save_frames",
    "crop_min_interval_seconds": "frame_min_interval_seconds",
}


# Removed along with the negative-veto mechanism. Warn rather than ignore --
# a config still saying "enabled: true" otherwise looks like it does something.
_REMOVED = (
    ("negative_suppression", None,
     "the negative-veto mechanism was removed; standing people are now rejected "
     "by geometry.min_aspect_ratio"),
    ("prompts", "negative_conf_floor",
     "negatives are distractor classes now and are never scored separately"),
)


def _warn_removed_keys(raw):
    for section, key, why in _REMOVED:
        if key is None:
            if section in raw:
                print(f"config: '{section}:' no longer does anything -- {why}. "
                      f"You can delete that section.")
        else:
            block = raw.get(section)
            if isinstance(block, dict) and key in block:
                print(f"config: '{section}.{key}' no longer does anything -- {why}. "
                      f"You can delete that line.")


def _migrate_legacy_keys(raw):
    out = raw.get("output")
    if not isinstance(out, dict):
        return
    for old, new in _RENAMED.items():
        if old in out and new not in out:
            out[new] = out.pop(old)
            print(f"config: output.{old} is now output.{new} -- using your value "
                  f"({out[new]!r}); rename it to silence this.")


def _validate(cfg, path):
    errors = []
    warnings = []

    if not cfg["prompts"]["positive"]:
        errors.append(
            "prompts.positive is empty. At least one positive prompt "
            "(e.g. 'a person lying on the floor') is required."
        )

    if not cfg["sources"]:
        errors.append(
            "sources is empty. Add at least one entry under 'sources:' "
            "(type: video_file / video_folder / rtsp)."
        )

    # secrets are expanded here, after defaults are merged and before any
    # source is opened, so an unset variable is reported with the others
    for src in cfg["sources"]:
        for key in ("url", "path"):
            if key in src:
                src[key] = _expand_env(src[key], f"source '{src.get('id', '?')}'.{key}",
                                       errors)

    valid_types = {"video_file", "video_folder", "rtsp"}
    seen_ids = set()
    for i, src in enumerate(cfg["sources"]):
        label = src.get("id", f"sources[{i}]")
        if "id" not in src:
            errors.append(f"sources[{i}] is missing 'id'.")
        elif src["id"] in seen_ids:
            errors.append(f"duplicate source id: {src['id']}")
        else:
            seen_ids.add(src["id"])

        if src.get("type") not in valid_types:
            errors.append(
                f"source '{label}': type must be one of {sorted(valid_types)}, "
                f"got {src.get('type')!r}"
            )

        if src.get("type") in ("video_file", "video_folder"):
            if not src.get("path"):
                errors.append(f"source '{label}': type={src.get('type')} requires 'path'")
            elif src.get("enabled", True) and not os.path.exists(src["path"]):
                # only enabled sources -- a disabled entry may legitimately point at
                # a path that isn't mounted right now
                errors.append(
                    f"source '{label}': path does not exist: {src['path']}"
                )

        mount = src.get("mount", "oblique")
        if mount not in ("oblique", "overhead"):
            errors.append(f"source '{label}': mount must be 'oblique' or "
                          f"'overhead', got {mount!r}")

        if src.get("type") == "rtsp" and not src.get("url"):
            errors.append(f"source '{label}': type=rtsp requires 'url'")

    if cfg["prompts"]["conf_threshold"] <= 0 or cfg["prompts"]["conf_threshold"] > 1:
        errors.append("prompts.conf_threshold must be in (0, 1]")

    vc = cfg["prompts"]["veto_conf"]
    if not isinstance(vc, (int, float)) or not (0 < vc <= 1):
        errors.append(f"prompts.veto_conf must be in (0, 1], got {vc!r}")
    vi = cfg["prompts"]["veto_iou"]
    if not isinstance(vi, (int, float)) or not (0 <= vi <= 1):
        errors.append(f"prompts.veto_iou must be in [0, 1], got {vi!r}")
    vk = cfg["prompts"]["veto_containment"]
    if not isinstance(vk, (int, float)) or not (0 < vk <= 1):
        errors.append(f"prompts.veto_containment must be in (0, 1], got {vk!r}")

    di = cfg["prompts"]["duplicate_iou"]
    if not isinstance(di, (int, float)) or not (0 < di <= 1):
        errors.append(f"prompts.duplicate_iou must be in (0, 1], got {di!r}")
    no = cfg["prompts"]["negative_override_iou"]
    if not isinstance(no, (int, float)) or not (0 <= no <= 1):
        errors.append(f"prompts.negative_override_iou must be in [0, 1] "
                      f"(0 disables), got {no!r}")
    elif 0 < no < 0.5:
        # the removed negative veto matched at 0.30 and deleted 39 genuine
        # man-downs, because a mattress under a fallen person overlaps them at
        # exactly that sort of value. Loosening this reintroduces that failure.
        warnings.append(
            f"prompts.negative_override_iou is {no}, low enough that a "
            f"distractor merely OVERLAPPING a person can delete them. The veto "
            f"removed from this project matched at 0.30 and erased 39 real "
            f"man-downs on crash-mat footage. Keep this at 0.75 or disable it "
            f"with 0."
        )

    T = cfg["temporal"]
    for key in ("confirm_frames", "window_frames"):
        if not isinstance(T[key], int) or T[key] < 1:
            errors.append(f"temporal.{key} must be a positive integer, got {T[key]!r}")
    if (isinstance(T["window_frames"], int) and isinstance(T["confirm_frames"], int)
            and T["window_frames"] < T["confirm_frames"]):
        errors.append(f"temporal.window_frames ({T['window_frames']}) must be >= "
                      f"temporal.confirm_frames ({T['confirm_frames']})")
    if not isinstance(T["match_iou"], (int, float)) or not (0 < T["match_iou"] <= 1):
        errors.append(f"temporal.match_iou must be in (0, 1], got {T['match_iou']!r}")

    S = cfg["stillness"]
    if S["still_seconds"] < 0:
        errors.append(f"stillness.still_seconds must be >= 0, got {S['still_seconds']!r}")
    for k in ("max_drift", "max_growth"):
        if not isinstance(S[k], (int, float)) or S[k] < 0:
            errors.append(f"stillness.{k} must be a non-negative number, got {S[k]!r}")
    if not (0 < S["match_iou"] <= 1):
        errors.append(f"stillness.match_iou must be in (0, 1], got {S['match_iou']!r}")
    if S["track_timeout_seconds"] <= 0:
        errors.append("stillness.track_timeout_seconds must be > 0")
    if not isinstance(S["min_observations"], int) or S["min_observations"] < 2:
        errors.append("stillness.min_observations must be an integer >= 2")

    layout = cfg["display"]["layout"]
    if layout not in ("separate", "grid"):
        errors.append(f"display.layout must be 'separate' or 'grid', got {layout!r}")

    margin = cfg["display"]["screen_margin"]
    if not isinstance(margin, (int, float)) or not (0 < margin <= 1):
        errors.append(f"display.screen_margin must be in (0, 1], got {margin!r}")
    for key in ("max_width", "max_height"):
        v = cfg["display"][key]
        if v is not None and (not isinstance(v, int) or v <= 0):
            errors.append(f"display.{key} must be a positive integer or null, got {v!r}")

    em = cfg["geometry"]["reject_edge_touching"]
    if em not in ("none", "bottom", "any"):
        errors.append(f"geometry.reject_edge_touching must be 'none', 'bottom' "
                      f"or 'any', got {em!r}")
    mg = cfg["geometry"]["edge_margin_px"]
    if not isinstance(mg, int) or mg < 0:
        errors.append(f"geometry.edge_margin_px must be a non-negative integer, got {mg!r}")

    ar = cfg["geometry"]["min_aspect_ratio"]
    if ar is not None and ar < 0:
        errors.append("geometry.min_aspect_ratio must be >= 0 (0 or null disables it)")

    # ---- BUG 8: the same camera entered under several ids ----
    seen_urls = {}
    for src in cfg["sources"]:
        if src.get("type") != "rtsp" or not src.get("enabled", True):
            continue
        u = src.get("url")
        if u:
            seen_urls.setdefault(u, []).append(src.get("id", "?"))
    for u, ids in seen_urls.items():
        if len(ids) > 1:
            host = u.split("@")[-1]
            warnings.append(
                f"sources {ids} are the same camera ({host}). Each would decode "
                f"the stream again, raise duplicate alarms for one incident, and "
                f"spend inference budget on copies. Give each id its own URL, or "
                f"disable the duplicates."
            )

    # ---- BUG 10: batch folders competing with live safety cameras ----
    live = [s.get("id") for s in cfg["sources"]
            if s.get("enabled", True) and s.get("type") == "rtsp"]
    batch = [s.get("id") for s in cfg["sources"]
             if s.get("enabled", True) and s.get("type") in ("video_folder", "video_file")]
    if live and batch:
        warnings.append(
            f"batch sources {batch} are enabled alongside live cameras {live}. "
            f"Backfill competes for the same serialized GPU as live monitoring and "
            f"forces the live display off. Run batch in a separate process."
        )

    # ---- BUG 9: throughput the deployment cannot meet ----
    #
    # The reference numbers below were measured on ONE machine (RTX 5050 Laptop,
    # 1920x1080 input). They are a starting point for whoever moves this to the
    # deployment GPU, not a fact about that GPU. Re-measure there:
    #
    #     python tools/bench_prompts.py --video <clip> --imgsz <n>
    #
    # and set logging.measured_inf_per_sec so this check uses the real number.
    #
    # This matters more than ordinary slowness: the stillness and flicker gates
    # count INFERENCE frames while still_seconds is wall-clock. If the GPU
    # cannot keep up, fewer inferences land inside the same 6 seconds, the gates
    # get harder to satisfy, and recall drops with nothing to announce it.
    n_live = len(live)
    if n_live:
        skip = max(1, cfg["processing"]["frame_skip"])
        reference = {("yolov8m-worldv2.pt", 960): 63.2,
                     ("yolov8m-worldv2.pt", 640): 85.5,
                     ("yolov8s-worldv2.pt", 960): 120.7,
                     ("yolov8s-worldv2.pt", 640): 210.8}
        measured = cfg["logging"].get("measured_inf_per_sec")
        have = measured or reference.get(
            (cfg["model"]["name"], cfg["model"]["image_size"]))
        if have:
            need = n_live * 15.0 / skip
            source = ("measured on this machine" if measured
                      else "reference figure from a different GPU -- RE-MEASURE")
            if need > have:
                warnings.append(
                    f"{n_live} live cameras at 15fps with frame_skip {skip} need "
                    f"~{need:.0f} inf/s; {cfg['model']['name']} at "
                    f"{cfg['model']['image_size']}px gives {have:.0f} inf/s "
                    f"({source}). Gates counted in inference frames get harder to "
                    f"satisfy as throughput falls, so recall degrades silently. "
                    f"Smaller model or image_size, higher frame_skip, or split "
                    f"across processes."
                )
            elif not measured:
                warnings.append(
                    "throughput headroom is being judged against a reference GPU, "
                    "not this one. Run tools/bench_prompts.py on the deployment "
                    "machine and set logging.measured_inf_per_sec."
                )

    # Deployment problems are shouted, not fatal: a safety system taken offline
    # by a config check is worse than one running with a known flaw. Correctness
    # errors below still refuse to start.
    for w in warnings:
        print(f"CONFIG WARNING: {w}")

    if errors:
        msg = f"Config errors in {path}:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigError(msg)


def enabled_sources(cfg):
    return [s for s in cfg["sources"] if s.get("enabled", True)]
