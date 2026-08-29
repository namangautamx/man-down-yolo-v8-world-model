# yw_mandown

YOLO-World man-down detection. One config-driven pipeline for video files,
video folders and live RTSP cameras.

## Setup

```bash
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml
# edit config/config.yaml: your sources, and read the comments before changing thresholds
```

The YOLO-World checkpoint downloads on first run.

## Run

```bash
# runs everything with enabled: true in config/config.yaml
python -m yw_mandown.cli --config config/config.yaml

# force specific sources, ignoring their enabled: flag
python -m yw_mandown.cli --config config/config.yaml --source cam1 --source cam2
```

One enabled source runs in the main thread and may show a live window. More
than one runs each on its own thread; the workers never call `cv2.imshow`
themselves -- they hand frames to the main thread, which owns every window.
Calling imshow from a worker is a real X11 hang, not a style preference.

Controls when a window is showing: `q` quits, `space` pauses.

## How a detection is decided

**YOLO-World is the only stage that detects anything. Every other stage only
removes candidates.** Nothing downstream can recover a fall the model never
proposed, so recall is capped at whatever `prompts.positive` emits above
`conf_threshold`; the remaining stages spend that recall to buy precision.

Per inference frame, in order ([pipeline.py](yw_mandown/pipeline.py)):

| # | stage | what it does | can it create a detection? |
|---|---|---|---|
| 1 | `YoloWorldDetector.infer` | proposes every candidate; boxes landing on a distractor class are dropped here | **yes -- the only one** |
| 2 | `filter_by_geometry` | drops boxes taller than wide, and boxes clipped by the frame edge | no |
| 3 | `conf_threshold` | drops low scores | no |
| 4 | `SecondPassVeto` | second model pass, animal classes; only runs when something already fired | no |
| 5 | `TemporalConfirmer` | flicker filter: needs `confirm_frames` hits within the last `window_frames` | no |
| 6 | `StillnessGate` | holds a box back until it proves it is not moving | no |

A detection that survives all six **is** the alarm. There is no hold: the
confirmer returns this frame's live detections or nothing, and never keeps an
alarm alive across a frame the model was silent on. An earlier version had a
hold that drew boxes the model never produced; it was removed for that reason.

Stages 5 and 6 count **inference** frames, so with `frame_skip: 2` a window of
4 spans 8 video frames.

### Why two different time-based gates

They do different jobs and both are needed.

The **flicker filter** is deliberately short. On a live cam1 run, 158 detected
frames were spread over 29 separate runs, 13 of them one frame long. That blink
is all this removes.

The **stillness gate** is the long one, and it is what separates a fallen person
from a crouching one. Of 19 false positives that survived every other filter, 14
were people genuinely not down -- crouching for a bag, kneeling by a car, bending
at a desk -- scoring 0.67, 0.70, 0.84. From a single box, crouching and lying are
not separable, so no prompt or threshold reaches them. Time does. Measured drift
per inference frame, in box widths:

```
genuinely fallen   0.0009  0.0012  0.0013  0.0015  0.0046
walking / active   0.0276  0.0320  0.0373
```

Thirty times apart, and normalised by box size so it is independent of resolution
and distance. **This delays every alarm by `still_seconds`.** That is the
mechanism, not a side effect -- you cannot know something is still without
watching it. It is paid once per event, not per frame.

The gate uses **median** drift across sightings, not a count of "still" frames:
counting flags gave a still-ratio of 0.58 on real falls and lost 7 of 8, because
the wobble is concentrated in a minority of frames.

## Watching it work

With `display.show_candidates: true`:

- **Yellow** -- passed the filters but not yet confirmed. The label carries the
  numbers that decide it: confidence, w/h, area %, stillness progress, median drift.
- **Red** -- confirmed man-down.

Only red is written to `events.jsonl` and saved as a frame. A yellow box is a
candidate under evaluation, not evidence. Set `show_candidates: false` for a
clean picture with confirmed alarms only.

## Output layout

```
<output.base_folder>/
  frames/<source_id>/   full annotated jpg per confirmed frame
                        (rate-limited by frame_min_interval_seconds)
  clips/<source_id>/    one mp4 per event, pre-roll + post-roll included
logs/
  events.jsonl          one line per confirmed detection
```

## Config reference

Every key is documented in `config/config.example.yaml`, usually with the
measurement that set it. The short version:

| Section | What it controls |
|---|---|
| `model` | checkpoint, `image_size`, device |
| `prompts.positive` | what counts as "man down" |
| `prompts.negative` | distractor classes; they absorb look-alike boxes and are never reported |
| `prompts.veto` | animal classes for the second pass, with `veto_conf` / `veto_iou` / `veto_containment` |
| `prompts.conf_threshold` | minimum confidence for a positive |
| `geometry` | `min_aspect_ratio` (reject tall boxes), `reject_edge_touching`, `edge_margin_px` |
| `temporal` | flicker filter: `confirm_frames` within `window_frames` |
| `stillness` | `still_seconds`, `max_drift`, `max_growth`, `match_iou`, `min_observations` |
| `processing.frame_skip` | run inference on every Nth frame; display and clips still get every frame |
| `output` | frame/clip saving, pre/post-roll seconds |
| `display` | window on/off, `layout`, screen fitting, `show_candidates` |
| `sources` | cameras and videos, each with `enabled: true/false` |

## Changing thresholds

**Thresholds are global. Do not tune them per camera.** A value fitted to one
camera is overfitting, and nothing that needs per-camera setup survives a
20-camera deployment. Validate a change across many cameras and keep one value.

Three traps, each learned the hard way and recorded in the config comments:

- **Adding a class to `prompts.negative` rescales every score.** YOLO-World
  compares a region against the whole class set, so a longer list shifts all
  confidences up. Adding "dog" and "cat" there took false positives on pre-fall
  frames from 1.9% to 24.4%. Compared at *matched* false-positive rates the
  shorter list won every time. This is why animal classes live in
  `prompts.veto` -- a separate pass -- instead.
- **Keep `prompts.veto` narrow and concrete.** "an animal lying down" scored
  0.28 on two real fallen construction workers and would have vetoed them.
- **Measure event-level recall, not frame-level.** Someone down for 10+ seconds
  produces many frames; missing some of them is not the same as missing the
  fall. Pick the operating point from false-positive count on a real, imbalanced
  set, not F1 on a balanced one.

If a source stops alarming, do not change thresholds blind:

```bash
python tools/diagnose_source.py --source cam2 --seconds 60
```

It reports every detection and the stage that rejected it, with the number
responsible (`aspect: w/h 0.64 < 0.7`), and saves annotated frames.

## Which numbers the measurements were taken at

`min_aspect_ratio` has been **0.7** in the shipped config since front-facing
falls were found to be rejected at 1.0 (a person lying along the camera axis
foreshortens to w/h 0.96). The earlier "92% of true detections kept at w/h >=
1.0" figure was measured at 1.0 and no longer describes the running pipeline.

0.7 admits a recorded standing false positive at w/h 0.91. That is a deliberate
trade: the same gate at 1.0 deleted a confirmed man-down. Both failures are now
recorded in `logs/audit.jsonl` rather than being invisible.

On `mount: overhead` sources the aspect gate is disabled entirely -- w/h is an
axis-aligned test and overhead body orientation is uniform over 360 degrees, so
the test does not merely weaken, it fires backwards.

## Benchmarking

Prompt count is close to free on a desktop GPU (RTX 5050: 3 classes 14.5 ms,
50 classes 14.8 ms) -- the CLIP text encoder runs once in `set_classes()` at
startup, not per frame. Other hardware differs, so measure on the target:

```bash
python tools/bench_prompts.py --video /path/to/clip.mp4 --imgsz 640
```

On Jetson run `sudo jetson_clocks` first; power mode moves latency far more than
prompt count does.

### Throughput, and what actually limits it

Measured on an RTX 5050 Laptop, 1920x1080 input:

```
model imgsz     ms   inf/s
    s   960    8.3   120.7
    s   640    4.7   210.8
    m   960   15.8    63.2
    m   640   11.7    85.5
```

Twenty cameras at 15 fps with `frame_skip: 2` needs ~150 inf/s.

The single model is shared across source threads behind a lock, and that lock
costs nothing: throughput is flat at ~120 inf/s from 1 thread to 20, and
batching does not help either (121 inf/s at batch 1 vs 114 at batch 20). One
960px forward pass already saturates the GPU, so **building a real batcher would
be wasted work**. The levers that move the ceiling are `model.name`,
`image_size`, `frame_skip`, and TensorRT.

Everything outside inference is negligible: all per-frame OpenCV work is 0.32 ms
at 1080p (~0.1 cores for 20 cameras) and CPU H.264 decode is ~1.2 ms/frame
(~0.36 cores). Note that `cv2` is CPU-only in the pip wheel -- no CUDA, no
GStreamer -- but that is not the bottleneck. On Jetson, use JetPack's system
OpenCV (which has GStreamer) with `nvv4l2decoder` to move decode onto NVDEC and
free CPU headroom.

## Qt font warnings

`QFontDatabase: Cannot find font directory .../cv2/qt/fonts` at startup is
cosmetic -- opencv-python ships Qt plugins but not Qt fonts. One-time fix:

```bash
python tools/fix_qt_fonts.py
```

Re-run after any `pip install --force-reinstall opencv-python`, which deletes
the directory again.
