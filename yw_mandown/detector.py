"""
Thin wrapper around ultralytics YOLOWorld.

On negative prompts
-------------------
This module used to run an OWLv2-style veto: negative prompts were detected at
their own confidence floor, and any positive box overlapping a negative box was
dropped. That mechanism is gone -- measured on the crash-mat footage it fired 39
times across 4 videos, every one of them from "a mattress", and every case
inspected was a genuine man-down. The model correctly finds the mat *underneath*
the fallen person and the veto then deleted the true detection.

The negative prompts themselves are still here and still matter. In YOLO-World
the class list IS the text-embedding set, so listing "a mattress" gives
mat-shaped boxes a class of their own to land in rather than competing with the
person boxes. Measured over the same 120 frames:

    negatives as classes, veto on    16/120 frames detected
    negatives as classes, veto off   17/120   <-- what this file now does
    negatives removed entirely       12/120

So they are set as classes and their boxes are then discarded. Deleting the
prompt list would cost about 30% of detections; deleting the veto cost nothing.

What separates a standing person from a fallen one is geometry, not class
overlap -- see filter_by_geometry below.
"""


def containment(inner, outer):
    """Fraction of `inner`'s area that lies inside `outer`.

    IoU alone misses the case that matters here: a man-down box drawn around a
    whole sofa scores low IoU against a small dog box sitting inside it, even
    though the dog IS the thing that triggered it. Containment catches that.
    """
    ax1, ay1, ax2, ay2 = inner
    bx1, by1, bx2, by2 = outer
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    return inter / area if area > 0 else 0.0


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union if union > 0 else 0.0


class Detection:
    __slots__ = ("xyxy", "conf", "class_name")

    def __init__(self, xyxy, conf, class_name):
        self.xyxy = xyxy  # (x1, y1, x2, y2) ints, clamped to frame
        self.conf = conf
        self.class_name = class_name



class YoloWorldDetector:
    def __init__(self, model_name, positive_prompts, negative_prompts,
                 image_size, device, conf_threshold):
        self.positive_prompts = list(positive_prompts)
        # set as classes to absorb look-alike boxes; never reported as detections
        self.negative_prompts = list(negative_prompts or [])
        self.all_prompts = self.positive_prompts + self.negative_prompts
        self.image_size = image_size
        self.device = device
        self.conf_threshold = conf_threshold

        from ultralytics import YOLOWorld  # imported here, not at module load,
                                            # so this module stays testable without torch

        print(f"Loading {model_name} ...")
        self.model = YOLOWorld(model_name)
        self.model.set_classes(self.all_prompts)
        print(f"Loaded. {len(self.positive_prompts)} positive prompts, "
              f"{len(self.negative_prompts)} distractor classes.")

    def infer(self, frame):
        """Returns the positive detections above conf_threshold, clamped to the
        frame. Boxes that land on a negative class are dropped here -- that is
        the whole point of listing them."""
        h, w = frame.shape[:2]
        results = self.model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        result = results[0]

        positives = []
        if result.boxes is None:
            return positives

        for box in result.boxes:
            class_id = int(box.cls[0].cpu().numpy())
            if class_id >= len(self.positive_prompts):
                continue  # a distractor class did its job; nothing to report

            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
            x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            positives.append(
                Detection((x1, y1, x2, y2), conf, self.all_prompts[class_id])
            )

        return positives

    @staticmethod
    def _clipped_by_edge(box, frame_shape, mode, margin):
        """Is this box cut off by the frame border?

        An aspect ratio only describes a pose when the whole body is inside the
        image. Someone walking towards the camera is clipped at the bottom, so
        only head and shoulders show -- wider than tall, and scored 0.97 as "a
        person lying on the floor" on cam2 (box [1052,981,1630,1440] in a
        1440-tall frame, y2 exactly on the border). The ratio was an artefact of
        the crop, not the person.

        Default is "bottom" because that is the walking-into-camera case. "any"
        also guards people clipped at the sides, at the risk of missing someone
        who really has fallen half out of view.
        """
        if mode == "none" or not frame_shape:
            return False
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        if y2 >= h - margin:                 # bottom, checked in both modes
            return True
        if mode == "any":
            return y1 <= margin or x1 <= margin or x2 >= w - margin
        return False

    @staticmethod
    def filter_by_geometry(positives, min_aspect_ratio, frame_shape=None,
                           edge_mode="none", edge_margin=8):
        """Drop positives whose box is taller than it is wide, or clipped by the
        frame edge.

        A person on the floor projects a wide box; a person standing projects a
        tall one. This is the single most reliable signal separating the two --
        far more so than confidence, because YOLO-World will happily score an
        upright person 0.55 on "a person lying on the floor" (measured on cam1
        at t=56.78s). Confidence cannot fix that; geometry can.

        The edge check exists because that reasoning only holds for a box that
        contains the whole body -- see _clipped_by_edge.
        """
        kept = []
        for p in positives:
            x1, y1, x2, y2 = p.xyxy
            h = y2 - y1
            if h <= 0:
                continue
            if min_aspect_ratio and (x2 - x1) / h < min_aspect_ratio:
                continue
            if YoloWorldDetector._clipped_by_edge(p.xyxy, frame_shape,
                                                  edge_mode, edge_margin):
                continue
            kept.append(p)
        return kept


class SecondPassVeto:
    """Veto a man-down detection when a specific object is really what is there.

    Why a SECOND model instead of extra classes on the first one: adding classes
    to the main class list rescales every score, because YOLO-World compares a
    region against the whole set. Putting "dog"/"cat" in prompts.negative took
    the empty-crash-mat box from 0.15 to 0.35 on an unchanged image and pushed
    false positives from 1.9% to 24.4%. A separate pass leaves the main scores
    exactly as tuned.

    Why it is affordable: this only runs on frames where a positive already
    survived, which is a few percent of frames. No detection, no cost.

    Why the class list must stay narrow: measured on real footage, "a dog" and
    "a cat" score 0.64-0.72 on actual animals but only 0.25-0.28 on genuinely
    fallen people. That margin is what makes the veto safe. A vaguer prompt like
    "an animal lying down" scored 0.28 on two real fallen construction workers
    and would have vetoed them -- do not add prompts of that kind here.
    """

    def __init__(self, model_name, veto_prompts, image_size, device,
                 veto_conf, veto_iou, veto_containment=0.60):
        self.prompts = list(veto_prompts or [])
        self.veto_conf = veto_conf
        self.veto_iou = veto_iou
        self.veto_containment = veto_containment
        self.model = None
        if not self.prompts:
            return

        from ultralytics import YOLOWorld
        self.image_size = image_size
        self.device = device
        print(f"Loading veto pass ({len(self.prompts)} classes: "
              f"{', '.join(self.prompts)}) ...")
        self.model = YOLOWorld(model_name)
        self.model.set_classes(self.prompts)

    @property
    def enabled(self):
        return self.model is not None

    def apply(self, frame, positives):
        """Drop positives whose box is better explained by a veto class."""
        if not self.enabled or not positives:
            return positives, []

        r = self.model.predict(frame, imgsz=self.image_size, conf=self.veto_conf,
                               device=self.device, verbose=False)[0]
        vetoes = []
        if r.boxes is not None:
            for b in r.boxes:
                conf = float(b.conf[0].cpu().numpy())
                x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                vetoes.append(Detection((x1, y1, x2, y2), conf,
                                        self.prompts[int(b.cls[0].cpu().numpy())]))
        if not vetoes:
            return positives, []

        # veto when the boxes match (IoU) OR when the veto object sits inside
        # the positive box (containment) -- see containment() for why both
        kept = [p for p in positives
                if not any(iou(p.xyxy, v.xyxy) >= self.veto_iou
                           or containment(v.xyxy, p.xyxy) >= self.veto_containment
                           for v in vetoes)]
        return kept, vetoes


# ---------------------------------------------------------------------------
