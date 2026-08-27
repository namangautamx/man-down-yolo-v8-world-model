# yw_mandown

YOLO-World man-down detection, combining your two standalone scripts
(video-folder batch test + live RTSP test) into one config-driven pipeline.

## What changed vs. the two original scripts

- One codebase, one entry point, both use cases (`video_file`, `video_folder`,
  `rtsp`) are just different `type:` values under `sources:` in config.
- **Event clips are now saved**, not just stills: a rolling pre-roll buffer
  plus a post-roll after the last detection, written to
  `output/clips/<source_id>/`.
- **Negative prompts are supported**, but work differently than in the OWLv2
  side of this project: YOLO-World gives one class per box, not a score per
  prompt per box, so there's no margin to subtract. Instead, a positive box
  listed as distractor classes so look-alike boxes land there instead of on
  the person; they never veto a detection. Standing people are rejected by
  default cost-wise -- only runs if `prompts.negative` is non-empty.
- **Multiple sources**: add as many `sources:` entries as you want. One
  enabled source runs in the main thread with the live window, matching your
  original scripts exactly. More than one enabled runs each on its own
  thread, with the live window forced off for all of them -- `cv2.imshow`
  from a worker thread is a real X11 hang/crash risk, not a style choice.
  With multiple sources, check `logs/events.jsonl` and the saved
  frames/clips instead of watching a window.
- Every detection is logged to `logs/events.jsonl` (source, frame, time,
  class, confidence, box).

## Setup

```bash
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml
# edit config/config.yaml: prompts, conf_threshold, and your sources
```

## Run

```bash
# runs whatever has enabled: true in config/config.yaml
python -m yw_mandown.cli --config config/config.yaml

# force just one source regardless of its enabled: flag
python -m yw_mandown.cli --config config/config.yaml --source cam1
```

Controls when a live window is showing: `q` quits that source, `space`
pauses/resumes.

## Output layout

```
output/
  frames/<source_id>/   one full annotated jpg per detected frame
                        (rate-limited by frame_min_interval_seconds)
  clips/<source_id>/    one mp4 per event, pre-roll + post-roll included
logs/
  events.jsonl          one line per detection
```

## Config reference

See the comments in `config/config.example.yaml` -- every key is documented
there. The short version:

| Section | What it controls |
|---|---|
| `model` | which YOLO-World checkpoint, image size, device |
| `prompts.positive` | what counts as "man down" |
| `prompts.negative` | distractor classes (mattress, standing person, ...); never veto |
| `prompts.conf_threshold` | minimum confidence to accept a positive detection |
| `geometry` | `min_aspect_ratio`: reject boxes taller than wide (standing people) |
| `temporal` | confirm/window/hold frames for a stable, non-flickering alarm |
| `processing.frame_skip` | run inference on every Nth frame |
| `output` | frame/clip saving, pre/post-roll seconds |
| `display` | live window on/off, `layout` (separate windows vs one grid), screen fitting |
| `sources` | your cameras/videos, each with its own `enabled: true/false` |

## Deploying to a new camera

YOLO-World reports a similarity between a region and a piece of text, not a
calibrated probability, so the same number means different things on different
cameras -- 0.25 was right on lab footage, 0.40 on an office camera. Do not
guess it. Measure it per camera:

```bash
```

Run it while the scene is normal -- people working and walking, nobody on the
floor. It records what the model scores on background and sets the threshold
just above that ceiling. Measured on cam2: background never exceeded 0.05,
against a configured threshold of 0.40 -- eight times more margin than needed,
which is why hard poses were being missed.

That is the deployment step for each new camera. One run, no hand-tuning.

If a source stops alarming, do not start changing thresholds blind:

```bash
python tools/diagnose_source.py --source cam2 --seconds 60
```

It reports every detection and the stage that rejected it, with the number
responsible (`aspect: w/h 0.74 < 1.0`), and saves annotated frames --
orange rejected, green passed.

## Qt font warnings

`QFontDatabase: Cannot find font directory .../cv2/qt/fonts` at startup is
cosmetic -- opencv-python ships Qt plugins but not Qt fonts. One-time fix:

```bash
python tools/fix_qt_fonts.py
```

Re-run it after any `pip install --force-reinstall opencv-python`, which
deletes the directory again. The app prints a one-line hint if it notices the
directory missing.

## Benchmarking prompt count

Prompt count is close to free on a desktop GPU (RTX 5050: 3 classes 14.5 ms,
50 classes 14.8 ms). Other hardware differs, so measure on the target device:

```bash
python tools/bench_prompts.py --video /path/to/clip.mp4
```

On Jetson run `sudo jetson_clocks` first -- power mode moves latency far more
than prompt count does.

## Notes carried over from the OWLv2 side of this project, still true here

- Pick the operating point from the false-positive count on a real,
  imbalanced eval set, not from F1 on a balanced one. A balanced smoke test
  flatters precision badly relative to what you'll see with 20 cameras
  running for hours.
- Frame-level recall understates event-level recall: someone down for 10+
  seconds gets many sampled frames, so missing some of them isn't the same
  as missing the fall. If you want a confirmation layer (N positive frames
  before alerting) rather than instant single-frame alerts, that's a
  reasonable next addition to `pipeline.py`, not currently built.
- `prompts.negative` earns its place as a class list, not as a veto: measured
  over 120 frames, keeping the entries with no veto detected 17 frames,
  the veto version 16, and deleting the entries entirely only 12.
