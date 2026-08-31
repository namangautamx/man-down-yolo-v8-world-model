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

One narrow exception was added later, in _dedupe: a positive IS dropped when a
distractor claims essentially the SAME box (IoU >= 0.75) AND outscores it. That
is not the old veto -- it fired at IoU 0.30 on any overlap, which is what let a
mattress under a fallen person delete the casualty. Read _dedupe for why the
two thresholds make that failure impossible here.

On duplicate boxes
------------------
ultralytics runs NMS per class, so classes never suppress each other and one
person yields one box per prompt that fires on them -- measured at 2.85 boxes
per person across logs/events.jsonl. _dedupe merges them.
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
                 image_size, device, conf_threshold,
                 duplicate_iou=0.60, negative_override_iou=0.75):
        self.positive_prompts = list(positive_prompts)
        # set as classes to absorb look-alike boxes; never reported as detections
        self.negative_prompts = list(negative_prompts or [])
        self.all_prompts = self.positive_prompts + self.negative_prompts
        self.image_size = image_size
        self.device = device
        self.conf_threshold = conf_threshold
        self.duplicate_iou = duplicate_iou
        self.negative_override_iou = negative_override_iou

        from ultralytics import YOLOWorld  # imported here, not at module load,
                                            # so this module stays testable without torch

        print(f"Loading {model_name} ...")
        self.model = YOLOWorld(model_name)
        self.model.set_classes(self.all_prompts)
        # Move the weights ONCE. ultralytics only honours `device` when it is
        # passed to predict(), and the model otherwise sits in system RAM --
        # measured 15.7ms/frame with `device: cuda` in the config and the
        # weights still on the CPU. set_classes() must come first: it attaches
        # the CLIP text encoder, and that has to move too.
        self._pin(device)
        print(f"Loaded. {len(self.positive_prompts)} positive prompts, "
              f"{len(self.negative_prompts)} distractor classes.")

    def _pin(self, device):
        """Put the weights on the target device once, and free the text encoder.

        The CLIP text encoder is only needed by set_classes(). Once the class
        embeddings are computed it is dead weight -- measured 164M parameters
        resident for a 13M detector, on every model instance including the veto
        pass. Dropping it is what brings the process back under the parameter
        budget.
        """
        import torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model.model.to(dev)
            self.model.overrides["device"] = dev
        except Exception as e:
            print(f"  could not pin weights to {dev}: {e}")
            return
        # the text encoder has served its purpose; release it
        freed = 0
        for attr in ("clip_model", "text_model", "txt_model"):
            m = getattr(self.model.model, attr, None)
            if m is not None:
                try:
                    freed += sum(p.numel() for p in m.parameters())
                except Exception:
                    pass
                setattr(self.model.model, attr, None)
        n = sum(p.numel() for p in self.model.model.parameters()) / 1e6
        print(f"  weights on {dev}, {n:.1f}M parameters"
              + (f" (released {freed/1e6:.0f}M text encoder)" if freed else ""))

    def infer(self, frame):
        """Returns (positives, negatives, shadows).

        positives   detections above conf_threshold, clamped to the frame
        negatives   distractor-class boxes; never reported, but needed by
                    dedupe(), so they are handed back rather than dropped here
        shadows     distractor boxes that claimed a region no positive is in

        Deduplication is deliberately NOT done here -- see dedupe().
        """
        h, w = frame.shape[:2]
        results = self.model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        result = results[0]

        positives, negatives = [], []
        if result.boxes is None:
            return positives, [], []

        for box in result.boxes:
            class_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
            x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            det = Detection((x1, y1, x2, y2), conf, self.all_prompts[class_id])
            (positives if class_id < len(self.positive_prompts)
             else negatives).append(det)

        # A distractor class winning the argmax normally means it did its job --
        # a mat-shaped box landed on "a mattress" instead of competing with the
        # person boxes. But the model also finds the mat UNDERNEATH a fallen
        # person: the removed veto fired 39 times on crash-mat footage and every
        # one was a real casualty. When a distractor box has NO positive box
        # overlapping it, that region was claimed entirely by the distractor and
        # a genuine man-down could have been erased with no record at all.
        #
        # Those are returned as "shadows" so the audit log can record them. They
        # are NOT promoted to detections -- doing so would undo the measured
        # benefit of the distractor classes (17/120 frames with them vs 12/120
        # without). Making the failure visible is the fix; guessing is not.
        shadows = [n for n in negatives
                   if not any(iou(n.xyxy, p.xyxy) >= 0.30 for p in positives)]
        return positives, negatives, shadows

    def dedupe(self, positives, negatives):
        """Merge boxes that describe the same object, and drop positives the
        model itself scores better as a distractor.

        MUST RUN AFTER THE GEOMETRY GATE, and the pipeline calls it there. Both
        rules below keep the highest-CONFIDENCE box of a group, and confidence
        says nothing about whether a box is shaped like a fallen person. Run
        before geometry, a fallen person carrying a good box at w/h 1.5 conf
        0.50 and a bad half-body box at w/h 0.6 conf 0.55 loses the good box to
        the bad one, which geometry then rejects -- turning two boxes on a real
        casualty into none. After geometry, only boxes that already passed the
        shape test compete, and the merge cannot lose a detection.

        Why this is needed at all: ultralytics runs NMS PER CLASS by default
        (`agnostic=False` offsets each class into its own coordinate space --
        ultralytics/utils/nms.py). Two classes therefore never suppress each
        other, so one person produces one box per prompt that fires on them.
        Measured on logs/events.jsonl: 112,876 alarm boxes over 39,539 distinct
        objects -- 2.85 boxes per person, 65% of every box drawn a duplicate of
        one already there.

        Two separate rules, with deliberately different thresholds:

        positive vs positive (duplicate_iou, default 0.60)
            "a person fallen" and "a person lying on the ground" landing on one
            body. Keep the highest-scoring box; the others say nothing new. This
            cannot change WHETHER a person is detected, only how many times.

        positive vs negative (negative_override_iou, default 0.75, and the
        negative must also score HIGHER)
            The model drew essentially the same box twice and preferred the
            distractor: "a person standing" 0.80 against "a person fallen" 0.45
            on one upright body. Currently the positive survives untouched,
            which is why standing people alarm.

            This is NOT the old negative veto, which was removed for good
            reason: that one fired at IoU 0.30 on ANY overlap and deleted 39
            genuine man-downs on crash-mat footage, because the model finds the
            mattress UNDERNEATH a fallen person. The two differences that make
            this safe are exactly the two that case failed:
              * 0.75 needs near-identical boxes. A mat box wraps the person and
                extends well past them, so it does not reach that overlap.
              * the distractor must WIN on confidence. On the crash-mat footage
                the mat scored 0.15-0.35 while the person scored higher, so it
                could not have overridden anything even at matching IoU.
            DEFAULT OFF (0.0), and the reason is a measurement, not caution.
            On the 24:58 casualty in eqiom_video, while he was flat on the
            ground, the model scored the SAME box:

                t=1496   "a person fallen" 0.167   "a person standing" 0.250
                t=1498   "a person fallen" 0.039   "a person standing" 0.355
                t=1500   "a person fallen" 0.091   "a person standing" 0.342

            The distractor won on every frame of a real man-down. Today the
            positive is already below conf_threshold so this rule changes
            nothing there -- but the obvious remedy for that miss is to lower
            conf_threshold, and this rule would then delete the casualty
            instead. That is the removed veto's failure with a new name.

            Enable it only after checking logs/audit.jsonl for stage "dedupe",
            reason "negative_override", against footage of real falls on YOUR
            cameras. Every drop is logged with the box that beat it.
        """
        dropped = []

        kept = []
        for p in sorted(positives, key=lambda d: -d.conf):
            dup = next((k for k in kept
                        if iou(p.xyxy, k.xyxy) >= self.duplicate_iou), None)
            if dup is None:
                kept.append(p)
            else:
                dropped.append((p, "duplicate_box", iou(p.xyxy, dup.xyxy)))

        if not self.negative_override_iou or not negatives:
            return kept, dropped

        survivors = []
        for p in kept:
            beat = next((n for n in negatives
                         if n.conf > p.conf
                         and iou(p.xyxy, n.xyxy) >= self.negative_override_iou),
                        None)
            if beat is None:
                survivors.append(p)
            else:
                dropped.append((p, f"negative_override:{beat.class_name}",
                                beat.conf))
        return survivors, dropped

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
        """Split positives into (kept, rejected).

        rejected carries (detection, reason, measured, limit) so the audit log
        can record the number responsible, not just that something vanished.

        A person on the floor projects a wide box; a person standing projects a
        tall one. This is the single most reliable signal separating the two --
        far more so than confidence, because YOLO-World will happily score an
        upright person 0.55 on "a person lying on the floor" (measured on cam1
        at t=56.78s). Confidence cannot fix that; geometry can.

        The edge check exists because that reasoning only holds for a box that
        contains the whole body -- see _clipped_by_edge.
        """
        kept, rejected = [], []
        for p in positives:
            x1, y1, x2, y2 = p.xyxy
            h = y2 - y1
            if h <= 0:
                rejected.append((p, "degenerate_box", 0.0, 0.0))
                continue
            ar = (x2 - x1) / h
            if min_aspect_ratio and ar < min_aspect_ratio:
                rejected.append((p, "aspect", ar, min_aspect_ratio))
                continue
            if YoloWorldDetector._clipped_by_edge(p.xyxy, frame_shape,
                                                  edge_mode, edge_margin):
                rejected.append((p, "frame_edge", float(y2), float(edge_margin)))
                continue
            kept.append(p)
        return kept, rejected


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
        YoloWorldDetector._pin(self, device)

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
