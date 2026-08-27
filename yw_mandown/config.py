"""
Config loading + validation.

Everything the pipeline does is driven from config/config.yaml. This module's
job is to load it, fill in defaults, and fail loudly (before any model is
loaded, before any camera is opened) if something required is missing or the
wrong type -- rather than crashing three minutes into a run.
"""

import os
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
    },
    "geometry": {
        "min_aspect_ratio": 1.0,        # 0 / null disables the check
        "reject_edge_touching": "bottom",  # "none" | "bottom" | "any"
        "edge_margin_px": 8,
    },
    "stillness": {
        "enabled": True,
        "still_seconds": 3.0,    # how long a box must hold position before it alarms
        "max_drift": 0.02,       # centroid movement per inference frame, in box widths
        "max_growth": 0.03,      # box area change per inference frame
        "match_iou": 0.30,       # IoU needed to call it the same box next frame
        "track_timeout_seconds": 2.0,
        "min_observations": 4,   # never confirm on fewer sightings than this
    },
    "temporal": {
        "enabled": True,
        "confirm_frames": 3,
        "window_frames": 6,
        "hold_frames": 15,
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

    t = cfg["temporal"]
    for key in ("confirm_frames", "window_frames", "hold_frames"):
        if not isinstance(t[key], int) or t[key] < 0:
            errors.append(f"temporal.{key} must be a non-negative integer, got {t[key]!r}")
    if isinstance(t["window_frames"], int) and isinstance(t["confirm_frames"], int):
        if t["window_frames"] < t["confirm_frames"]:
            errors.append(
                f"temporal.window_frames ({t['window_frames']}) must be >= "
                f"temporal.confirm_frames ({t['confirm_frames']})"
            )
    if isinstance(t["confirm_frames"], int) and t["confirm_frames"] < 1:
        errors.append("temporal.confirm_frames must be >= 1")

    if errors:
        msg = f"Config errors in {path}:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigError(msg)


def enabled_sources(cfg):
    return [s for s in cfg["sources"] if s.get("enabled", True)]
