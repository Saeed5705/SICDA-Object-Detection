"""
SICDA Detection Evaluator
=========================

Purpose:
    Robust evaluation of SICDA RGB-Thermal object detection.

Classes:
    0 -> background
    1 -> person
    2 -> car
    3 -> bike

Metrics:
    AP50
    AP75
    AP95
    mAP50:95
    Precision
    Recall
    F1-score
    TP / FP / FN

Important:
    1. AP uses ALL valid predictions ranked by confidence.
       No score threshold is applied to AP.

    2. Precision / Recall / F1 uses score_threshold.

    3. Matching is:
           class-aware
           one-to-one
           IoU-based

    4. Background is NEVER evaluated as an object class.

    5. Invalid boxes / labels / scores are safely removed.

    6. Empty predictions and empty targets are handled safely.

    7. Per-class detection statistics are reported.

    8. Complexity statistics are reported when a model is supplied.
"""

import os
import numpy as np
import torch


# ======================================================================
# SICDA EVALUATOR
# ======================================================================

class Evaluator:

    def __init__(
        self,
        num_classes=4,
        iou_threshold=0.50,
        score_threshold=0.05,
        model=None,
        device="cuda",
        save_dir="./outputs",
    ):

        # ==============================================================
        # Configuration
        # ==============================================================

        self.num_classes = int(num_classes)

        self.iou_threshold = float(
            iou_threshold
        )

        self.score_threshold = float(
            score_threshold
        )

        self.model = model

        self.device = device

        self.save_dir = save_dir

        os.makedirs(
            self.save_dir,
            exist_ok=True
        )

        # ==============================================================
        # Class names
        # ==============================================================

        self.class_names = {

            0: "background",
            1: "person",
            2: "car",
            3: "bike",

        }

        # ==============================================================
        # COCO-style IoU thresholds
        #
        # 0.50, 0.55, ..., 0.95
        # ==============================================================

        self.iou_thresholds = np.arange(
            0.50,
            0.951,
            0.05,
            dtype=np.float64
        )

        # ==============================================================
        # Reset
        # ==============================================================

        self.reset()

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(self):

        self.predictions = []

        self.targets = []

        # AP
        self.class_AP = {}
        self.class_AP75 = {}
        self.class_AP95 = {}
        self.class_AP_coco = {}

        self.map50 = 0.0
        self.map75 = 0.0
        self.map95 = 0.0
        self.map = 0.0

        # Detection metrics
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0

        self.tp = 0
        self.fp = 0
        self.fn = 0

        # Dataset statistics
        self.num_images = 0
        self.total_gt_boxes = 0
        self.total_pred_boxes = 0

        # Per-class statistics
        self.class_gt_count = {}
        self.class_pred_count = {}
        self.class_tp = {}
        self.class_fp = {}
        self.class_fn = {}

    # ==================================================================
    # TENSOR CONVERSION
    # ==================================================================

    @staticmethod
    def _to_cpu_tensor(
        value,
        dtype
    ):

        if value is None:

            return torch.empty(
                0,
                dtype=dtype
            )

        if torch.is_tensor(value):

            return (
                value
                .detach()
                .cpu()
                .to(dtype)
            )

        return torch.as_tensor(
            value,
            dtype=dtype
        )

    # ==================================================================
    # SANITIZE BOXES
    # ==================================================================

    @staticmethod
    def _sanitize_boxes(
        boxes
    ):

        boxes = Evaluator._to_cpu_tensor(
            boxes,
            torch.float32
        )

        # --------------------------------------------------------------
        # Empty
        # --------------------------------------------------------------

        if boxes.numel() == 0:

            return torch.empty(
                (0, 4),
                dtype=torch.float32
            )

        # --------------------------------------------------------------
        # Reshape
        # --------------------------------------------------------------

        boxes = boxes.reshape(
            -1,
            4
        )

        # --------------------------------------------------------------
        # Remove NaN / Inf
        # --------------------------------------------------------------

        finite = torch.isfinite(
            boxes
        ).all(
            dim=1
        )

        boxes = boxes[
            finite
        ]

        if len(boxes) == 0:

            return torch.empty(
                (0, 4),
                dtype=torch.float32
            )

        # --------------------------------------------------------------
        # Ensure x1 <= x2 and y1 <= y2
        # --------------------------------------------------------------

        x1 = torch.minimum(
            boxes[:, 0],
            boxes[:, 2]
        )

        y1 = torch.minimum(
            boxes[:, 1],
            boxes[:, 3]
        )

        x2 = torch.maximum(
            boxes[:, 0],
            boxes[:, 2]
        )

        y2 = torch.maximum(
            boxes[:, 1],
            boxes[:, 3]
        )

        boxes = torch.stack(
            [
                x1,
                y1,
                x2,
                y2
            ],
            dim=1
        )

        # --------------------------------------------------------------
        # Remove zero-area boxes
        # --------------------------------------------------------------

        valid = (
            (boxes[:, 2] > boxes[:, 0])
            &
            (boxes[:, 3] > boxes[:, 1])
        )

        boxes = boxes[
            valid
        ]

        return boxes

    # ==================================================================
    # SANITIZE PREDICTION
    # ==================================================================

    def _sanitize_prediction(
        self,
        detection
    ):

        if not isinstance(
            detection,
            dict
        ):

            return {
                "boxes": torch.empty(
                    (0, 4)
                ),
                "labels": torch.empty(
                    (0,),
                    dtype=torch.long
                ),
                "scores": torch.empty(
                    (0,)
                )
            }

        boxes = self._sanitize_boxes(
            detection.get(
                "boxes",
                torch.empty((0, 4))
            )
        )

        labels = self._to_cpu_tensor(
            detection.get(
                "labels",
                torch.empty(
                    (0,),
                    dtype=torch.long
                )
            ),
            torch.long
        ).reshape(-1)

        scores = self._to_cpu_tensor(
            detection.get(
                "scores",
                torch.empty((0,))
            ),
            torch.float32
        ).reshape(-1)

        # --------------------------------------------------------------
        # Keep equal number of entries
        # --------------------------------------------------------------

        n = min(
            len(boxes),
            len(labels),
            len(scores)
        )

        boxes = boxes[:n]
        labels = labels[:n]
        scores = scores[:n]

        if n == 0:

            return {
                "boxes": torch.empty(
                    (0, 4)
                ),
                "labels": torch.empty(
                    (0,),
                    dtype=torch.long
                ),
                "scores": torch.empty(
                    (0,)
                )
            }

        # --------------------------------------------------------------
        # Remove invalid scores
        # --------------------------------------------------------------

        valid = torch.isfinite(
            scores
        )

        # --------------------------------------------------------------
        # Remove invalid labels
        # --------------------------------------------------------------

        valid = (
            valid
            &
            (labels >= 1)
            &
            (labels < self.num_classes)
        )

        boxes = boxes[
            valid
        ]

        labels = labels[
            valid
        ]

        scores = scores[
            valid
        ]

        # --------------------------------------------------------------
        # Sort by confidence
        # --------------------------------------------------------------

        if len(scores) > 0:

            order = torch.argsort(
                scores,
                descending=True
            )

            boxes = boxes[
                order
            ]

            labels = labels[
                order
            ]

            scores = scores[
                order
            ]

        return {
            "boxes": boxes,
            "labels": labels,
            "scores": scores
        }

    # ==================================================================
    # SANITIZE TARGET
    # ==================================================================

    def _sanitize_target(
        self,
        target
    ):

        if not isinstance(
            target,
            dict
        ):

            return {
                "boxes": torch.empty(
                    (0, 4)
                ),
                "labels": torch.empty(
                    (0,),
                    dtype=torch.long
                )
            }

        boxes = self._sanitize_boxes(
            target.get(
                "boxes",
                torch.empty((0, 4))
            )
        )

        labels = self._to_cpu_tensor(
            target.get(
                "labels",
                torch.empty(
                    (0,),
                    dtype=torch.long
                )
            ),
            torch.long
        ).reshape(-1)

        n = min(
            len(boxes),
            len(labels)
        )

        boxes = boxes[:n]
        labels = labels[:n]

        # --------------------------------------------------------------
        # Remove background and invalid labels
        # --------------------------------------------------------------

        valid = (
            (labels >= 1)
            &
            (labels < self.num_classes)
        )

        boxes = boxes[
            valid
        ]

        labels = labels[
            valid
        ]

        return {
            "boxes": boxes,
            "labels": labels
        }

    # ==================================================================
    # IOU
    # ==================================================================

    @staticmethod
    def compute_iou(
        box1,
        box2
    ):

        if torch.is_tensor(box1):

            box1 = (
                box1
                .detach()
                .cpu()
                .float()
            )

        else:

            box1 = torch.as_tensor(
                box1,
                dtype=torch.float32
            )

        if torch.is_tensor(box2):

            box2 = (
                box2
                .detach()
                .cpu()
                .float()
            )

        else:

            box2 = torch.as_tensor(
                box2,
                dtype=torch.float32
            )

        # --------------------------------------------------------------
        # Intersection
        # --------------------------------------------------------------

        x1 = max(
            float(box1[0]),
            float(box2[0])
        )

        y1 = max(
            float(box1[1]),
            float(box2[1])
        )

        x2 = min(
            float(box1[2]),
            float(box2[2])
        )

        y2 = min(
            float(box1[3]),
            float(box2[3])
        )

        inter_w = max(
            0.0,
            x2 - x1
        )

        inter_h = max(
            0.0,
            y2 - y1
        )

        intersection = (
            inter_w
            *
            inter_h
        )

        # --------------------------------------------------------------
        # Areas
        # --------------------------------------------------------------

        area1 = max(
            0.0,
            float(box1[2] - box1[0])
        ) * max(
            0.0,
            float(box1[3] - box1[1])
        )

        area2 = max(
            0.0,
            float(box2[2] - box2[0])
        ) * max(
            0.0,
            float(box2[3] - box2[1])
        )

        union = (
            area1
            +
            area2
            -
            intersection
        )

        if union <= 0.0:

            return 0.0

        return float(
            intersection / union
        )

    # ==================================================================
    # UPDATE
    # ==================================================================

    def update(
        self,
        detections,
        batch
    ):

        """
        Add validation results.

        Expected batch:

            {
                "targets": list[dict]
            }

        Expected detection:

            {
                "boxes": Tensor[N,4],
                "labels": Tensor[N],
                "scores": Tensor[N]
            }
        """

        if not isinstance(
            batch,
            dict
        ):

            raise TypeError(
                "Evaluator expected batch as dict, "
                f"got {type(batch)}"
            )

        if "targets" not in batch:

            raise KeyError(
                "Evaluator batch does not contain "
                "'targets'."
            )

        targets = batch["targets"]

        if targets is None:

            targets = []

        if detections is None:

            detections = []

        if not isinstance(
            detections,
            (list, tuple)
        ):

            raise TypeError(
                "Detections must be list/tuple."
            )

        if not isinstance(
            targets,
            (list, tuple)
        ):

            raise TypeError(
                "Targets must be list/tuple."
            )

        # --------------------------------------------------------------
        # Count
        # --------------------------------------------------------------

        count = min(
            len(detections),
            len(targets)
        )

        if len(detections) != len(targets):

            print(
                "[WARNING] Detection/target count mismatch: "
                f"{len(detections)} vs {len(targets)}"
            )

        # --------------------------------------------------------------
        # Store
        # --------------------------------------------------------------

        for index in range(count):

            prediction = (
                self._sanitize_prediction(
                    detections[index]
                )
            )

            target = (
                self._sanitize_target(
                    targets[index]
                )
            )

            self.predictions.append(
                prediction
            )

            self.targets.append(
                target
            )

            self.num_images += 1

            self.total_pred_boxes += len(
                prediction["boxes"]
            )

            self.total_gt_boxes += len(
                target["boxes"]
            )

            # ----------------------------------------------------------
            # Per-class counts
            # ----------------------------------------------------------

            for class_id in range(
                1,
                self.num_classes
            ):

                gt_count = int(
                    (
                        target["labels"]
                        ==
                        class_id
                    ).sum().item()
                )

                pred_count = int(
                    (
                        prediction["labels"]
                        ==
                        class_id
                    ).sum().item()
                )

                self.class_gt_count[class_id] = (
                    self.class_gt_count.get(
                        class_id,
                        0
                    )
                    +
                    gt_count
                )

                self.class_pred_count[class_id] = (
                    self.class_pred_count.get(
                        class_id,
                        0
                    )
                    +
                    pred_count
                )

    # ==================================================================
    # PRECISION-RECALL INTEGRATION
    # ==================================================================

    @staticmethod
    def _integrate_pr(
        recall,
        precision
    ):

        if recall.size == 0:

            return 0.0

        mrec = np.concatenate(
            (
                [0.0],
                recall,
                [1.0]
            )
        )

        mpre = np.concatenate(
            (
                [0.0],
                precision,
                [0.0]
            )
        )

        # --------------------------------------------------------------
        # Precision envelope
        # --------------------------------------------------------------

        for i in range(
            len(mpre) - 1,
            0,
            -1
        ):

            mpre[i - 1] = max(
                mpre[i - 1],
                mpre[i]
            )

        # --------------------------------------------------------------
        # Recall changes
        # --------------------------------------------------------------

        changing = np.where(
            mrec[1:] != mrec[:-1]
        )[0]

        if len(changing) == 0:

            return 0.0

        return float(
            np.sum(
                (
                    mrec[changing + 1]
                    -
                    mrec[changing]
                )
                *
                mpre[changing + 1]
            )
        )

    # ==================================================================
    # AP AT ONE IOU
    # ==================================================================

    def compute_ap_at_iou(
        self,
        class_id,
        threshold
    ):

        """
        Compute AP for one class at one IoU threshold.

        NO confidence threshold is used.

        All predictions are ranked by confidence.
        """

        total_gt = 0

        gt_by_image = {}

        matched_by_image = {}

        predictions = []

        # --------------------------------------------------------------
        # Collect GT and predictions
        # --------------------------------------------------------------

        for image_idx, (
            prediction,
            target
        ) in enumerate(
            zip(
                self.predictions,
                self.targets
            )
        ):

            gt_mask = (
                target["labels"]
                ==
                class_id
            )

            gt_boxes = (
                target["boxes"][
                    gt_mask
                ]
            )

            gt_by_image[
                image_idx
            ] = gt_boxes

            matched_by_image[
                image_idx
            ] = torch.zeros(
                len(gt_boxes),
                dtype=torch.bool
            )

            total_gt += len(
                gt_boxes
            )

            pred_mask = (
                prediction["labels"]
                ==
                class_id
            )

            pred_boxes = (
                prediction["boxes"][
                    pred_mask
                ]
            )

            pred_scores = (
                prediction["scores"][
                    pred_mask
                ]
            )

            for box, score in zip(
                pred_boxes,
                pred_scores
            ):

                predictions.append(
                    (
                        float(score),
                        image_idx,
                        box
                    )
                )

        # --------------------------------------------------------------
        # No GT
        # --------------------------------------------------------------

        if total_gt == 0:

            return 0.0

        # --------------------------------------------------------------
        # No predictions
        # --------------------------------------------------------------

        if len(predictions) == 0:

            return 0.0

        # --------------------------------------------------------------
        # Confidence ranking
        # --------------------------------------------------------------

        predictions.sort(
            key=lambda item: item[0],
            reverse=True
        )

        tp = np.zeros(
            len(predictions),
            dtype=np.float64
        )

        fp = np.zeros(
            len(predictions),
            dtype=np.float64
        )

        # --------------------------------------------------------------
        # One-to-one matching
        # --------------------------------------------------------------

        for pred_idx, (
            score,
            image_idx,
            box
        ) in enumerate(
            predictions
        ):

            gt_boxes = gt_by_image[
                image_idx
            ]

            matched = matched_by_image[
                image_idx
            ]

            best_iou = 0.0
            best_gt = -1

            for gt_idx, gt_box in enumerate(
                gt_boxes
            ):

                if matched[
                    gt_idx
                ]:

                    continue

                iou = self.compute_iou(
                    box,
                    gt_box
                )

                if iou > best_iou:

                    best_iou = iou
                    best_gt = gt_idx

            if (
                best_gt >= 0
                and
                best_iou >= threshold
            ):

                tp[
                    pred_idx
                ] = 1.0

                matched[
                    best_gt
                ] = True

            else:

                fp[
                    pred_idx
                ] = 1.0

        # --------------------------------------------------------------
        # PR curve
        # --------------------------------------------------------------

        tp_cum = np.cumsum(
            tp
        )

        fp_cum = np.cumsum(
            fp
        )

        recall = (
            tp_cum
            /
            max(
                float(total_gt),
                1e-12
            )
        )

        precision = (
            tp_cum
            /
            np.maximum(
                tp_cum + fp_cum,
                1e-12
            )
        )

        return self._integrate_pr(
            recall,
            precision
        )

    # ==================================================================
    # COMPUTE MAP
    # ==================================================================

    def compute_map(self):

        self.class_AP = {}
        self.class_AP75 = {}
        self.class_AP95 = {}
        self.class_AP_coco = {}

        ap50_values = []
        ap75_values = []
        ap95_values = []
        coco_values_all = []

        # --------------------------------------------------------------
        # Object classes only
        # --------------------------------------------------------------

        for class_id in range(
            1,
            self.num_classes
        ):

            ap50 = self.compute_ap_at_iou(
                class_id,
                0.50
            )

            ap75 = self.compute_ap_at_iou(
                class_id,
                0.75
            )

            ap95 = self.compute_ap_at_iou(
                class_id,
                0.95
            )

            class_coco = []

            for threshold in self.iou_thresholds:

                class_coco.append(
                    self.compute_ap_at_iou(
                        class_id,
                        float(threshold)
                    )
                )

            ap_coco = float(
                np.mean(
                    class_coco
                )
            )

            self.class_AP[
                class_id
            ] = float(ap50)

            self.class_AP75[
                class_id
            ] = float(ap75)

            self.class_AP95[
                class_id
            ] = float(ap95)

            self.class_AP_coco[
                class_id
            ] = float(ap_coco)

            ap50_values.append(
                ap50
            )

            ap75_values.append(
                ap75
            )

            ap95_values.append(
                ap95
            )

            coco_values_all.append(
                ap_coco
            )

        self.map50 = (
            float(np.mean(ap50_values))
            if ap50_values
            else 0.0
        )

        self.map75 = (
            float(np.mean(ap75_values))
            if ap75_values
            else 0.0
        )

        self.map95 = (
            float(np.mean(ap95_values))
            if ap95_values
            else 0.0
        )

        self.map = (
            float(np.mean(coco_values_all))
            if coco_values_all
            else 0.0
        )

    # ==================================================================
    # PRECISION / RECALL / F1
    # ==================================================================

    def compute_precision_recall(self):

        tp = 0
        fp = 0
        fn = 0

        self.class_tp = {}
        self.class_fp = {}
        self.class_fn = {}

        # --------------------------------------------------------------
        # Initialize per-class statistics
        # --------------------------------------------------------------

        for class_id in range(
            1,
            self.num_classes
        ):

            self.class_tp[class_id] = 0
            self.class_fp[class_id] = 0
            self.class_fn[class_id] = 0

        # --------------------------------------------------------------
        # Image loop
        # --------------------------------------------------------------

        for prediction, target in zip(
            self.predictions,
            self.targets
        ):

            gt_boxes = target[
                "boxes"
            ]

            gt_labels = target[
                "labels"
            ]

            # ----------------------------------------------------------
            # Confidence threshold
            # ----------------------------------------------------------

            keep = (
                prediction["scores"]
                >=
                self.score_threshold
            )

            pred_boxes = (
                prediction["boxes"][
                    keep
                ]
            )

            pred_labels = (
                prediction["labels"][
                    keep
                ]
            )

            pred_scores = (
                prediction["scores"][
                    keep
                ]
            )

            # ----------------------------------------------------------
            # Sort by confidence
            # ----------------------------------------------------------

            if len(pred_scores) > 0:

                order = torch.argsort(
                    pred_scores,
                    descending=True
                )

                pred_boxes = pred_boxes[
                    order
                ]

                pred_labels = pred_labels[
                    order
                ]

            matched = torch.zeros(
                len(gt_boxes),
                dtype=torch.bool
            )

            # ----------------------------------------------------------
            # Prediction matching
            # ----------------------------------------------------------

            for box, label in zip(
                pred_boxes,
                pred_labels
            ):

                label_id = int(
                    label.item()
                )

                best_iou = 0.0
                best_gt = -1

                for gt_idx, gt_box in enumerate(
                    gt_boxes
                ):

                    if matched[
                        gt_idx
                    ]:

                        continue

                    if int(
                        gt_labels[
                            gt_idx
                        ].item()
                    ) != label_id:

                        continue

                    iou = self.compute_iou(
                        box,
                        gt_box
                    )

                    if iou > best_iou:

                        best_iou = iou
                        best_gt = gt_idx

                if (
                    best_gt >= 0
                    and
                    best_iou >= self.iou_threshold
                ):

                    matched[
                        best_gt
                    ] = True

                    tp += 1

                    self.class_tp[
                        label_id
                    ] += 1

                else:

                    fp += 1

                    self.class_fp[
                        label_id
                    ] += 1

            # ----------------------------------------------------------
            # False negatives
            # ----------------------------------------------------------

            for gt_idx, gt_label in enumerate(
                gt_labels
            ):

                if not matched[
                    gt_idx
                ]:

                    fn += 1

                    label_id = int(
                        gt_label.item()
                    )

                    if label_id in self.class_fn:

                        self.class_fn[
                            label_id
                        ] += 1

        precision = (
            tp /
            max(
                tp + fp,
                1e-12
            )
        )

        recall = (
            tp /
            max(
                tp + fn,
                1e-12
            )
        )

        f1 = (
            2.0
            *
            precision
            *
            recall
            /
            max(
                precision + recall,
                1e-12
            )
        )

        return (
            float(precision),
            float(recall),
            float(f1),
            int(tp),
            int(fp),
            int(fn)
        )

    # ==================================================================
    # PER-CLASS DETECTION STATISTICS
    # ==================================================================

    def print_detection_class_statistics(self):

        print()

        print(
            "CLASS-WISE DETECTION STATISTICS"
        )

        print(
            "=" * 82
        )

        print(
            f"{'Class':<15}"
            f"{'GT':>10}"
            f"{'Pred':>10}"
            f"{'TP':>10}"
            f"{'FP':>10}"
            f"{'FN':>10}"
        )

        print(
            "-" * 82
        )

        for class_id in range(
            1,
            self.num_classes
        ):

            name = self.class_names.get(
                class_id,
                f"class_{class_id}"
            )

            print(
                f"{name:<15}"
                f"{self.class_gt_count.get(class_id, 0):>10}"
                f"{self.class_pred_count.get(class_id, 0):>10}"
                f"{self.class_tp.get(class_id, 0):>10}"
                f"{self.class_fp.get(class_id, 0):>10}"
                f"{self.class_fn.get(class_id, 0):>10}"
            )

        print(
            "=" * 82
        )

    # ==================================================================
    # DATASET SUMMARY
    # ==================================================================

    def print_dataset_summary(self):

        print()

        print(
            "VALIDATION DATA SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            f"Validation images      : "
            f"{self.num_images}"
        )

        print(
            f"Ground-truth boxes     : "
            f"{self.total_gt_boxes}"
        )

        print(
            f"Predicted boxes        : "
            f"{self.total_pred_boxes}"
        )

        print(
            "=" * 70
        )

    # ==================================================================
    # COMPLEXITY ANALYSIS
    # ==================================================================

    def complexity_analysis(self):

        if self.model is None:

            print()

            print(
                "Computational Complexity"
            )

            print(
                "=" * 60
            )

            print(
                "Model               : Not supplied"
            )

            return

        params = sum(
            p.numel()
            for p in self.model.parameters()
        )

        trainable = sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )

        size_mb = (
            params
            *
            4
            /
            (1024 ** 2)
        )

        print()

        print(
            "Computational Complexity"
        )

        print(
            "=" * 60
        )

        print(
            f"Parameters       : "
            f"{params / 1e6:.2f} M"
        )

        print(
            f"Trainable Params : "
            f"{trainable / 1e6:.2f} M"
        )

        print(
            f"Model Size (FP32): "
            f"{size_mb:.2f} MB"
        )

        if torch.cuda.is_available():

            print(
                f"GPU Memory       : "
                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )

        print(
            "=" * 60
        )

    # ==================================================================
    # PRINT CLASS AP RESULTS
    # ==================================================================

    def print_class_results(self):

        print()

        print(
            "CLASS-WISE AP RESULTS"
        )

        print(
            "=" * 82
        )

        print(
            f"{'Class':<15}"
            f"{'AP50':>12}"
            f"{'AP75':>12}"
            f"{'AP95':>12}"
            f"{'AP50:95':>15}"
        )

        print(
            "-" * 82
        )

        for class_id in range(
            1,
            self.num_classes
        ):

            class_name = self.class_names.get(
                class_id,
                f"class_{class_id}"
            )

            print(
                f"{class_name:<15}"
                f"{self.class_AP.get(class_id, 0.0):>12.6f}"
                f"{self.class_AP75.get(class_id, 0.0):>12.6f}"
                f"{self.class_AP95.get(class_id, 0.0):>12.6f}"
                f"{self.class_AP_coco.get(class_id, 0.0):>15.6f}"
            )

        print(
            "-" * 82
        )

        print(
            f"{'Mean':<15}"
            f"{self.map50:>12.6f}"
            f"{self.map75:>12.6f}"
            f"{self.map95:>12.6f}"
            f"{self.map:>15.6f}"
        )

        print(
            "=" * 82
        )

    # ==================================================================
    # EVALUATE
    # ==================================================================

    def evaluate(self):

        # --------------------------------------------------------------
        # AP / mAP
        # --------------------------------------------------------------

        self.compute_map()

        # --------------------------------------------------------------
        # Precision / Recall / F1
        # --------------------------------------------------------------

        (
            precision,
            recall,
            f1,
            tp,
            fp,
            fn
        ) = self.compute_precision_recall()

        self.precision = precision
        self.recall = recall
        self.f1_score = f1

        self.tp = tp
        self.fp = fp
        self.fn = fn

        # --------------------------------------------------------------
        # Dataset summary
        # --------------------------------------------------------------

        self.print_dataset_summary()

        # --------------------------------------------------------------
        # Detection summary
        # --------------------------------------------------------------

        print()

        print(
            "Detection Summary @ IoU "
            f"{self.iou_threshold:.2f}"
        )

        print(
            "=" * 70
        )

        print(
            f"Score Threshold : "
            f"{self.score_threshold:.3f}"
        )

        print(
            f"TP / FP / FN    : "
            f"{tp} / {fp} / {fn}"
        )

        print(
            f"Precision       : "
            f"{precision:.4f}"
        )

        print(
            f"Recall          : "
            f"{recall:.4f}"
        )

        print(
            f"F1-score        : "
            f"{f1:.4f}"
        )

        print(
            "=" * 70
        )

        # --------------------------------------------------------------
        # Complexity
        # --------------------------------------------------------------

        self.complexity_analysis()

        # --------------------------------------------------------------
        # AP
        # --------------------------------------------------------------

        self.print_class_results()

        # --------------------------------------------------------------
        # Per-class detection statistics
        # --------------------------------------------------------------

        self.print_detection_class_statistics()

        # --------------------------------------------------------------
        # Final
        # --------------------------------------------------------------

        print()

        print(
            "FINAL DETECTION METRICS"
        )

        print(
            "=" * 70
        )

        print(
            f"mAP50      : "
            f"{self.map50:.6f}"
        )

        print(
            f"mAP75      : "
            f"{self.map75:.6f}"
        )

        print(
            f"mAP95      : "
            f"{self.map95:.6f}"
        )

        print(
            f"mAP50:95   : "
            f"{self.map:.6f}"
        )

        print(
            f"Precision  : "
            f"{precision:.6f}"
        )

        print(
            f"Recall     : "
            f"{recall:.6f}"
        )

        print(
            f"F1-score   : "
            f"{f1:.6f}"
        )

        print(
            "=" * 70
        )

        # --------------------------------------------------------------
        # Return dictionary
        # --------------------------------------------------------------

        return {

            # Global metrics
            "mAP":
                self.map,

            "mAP50":
                self.map50,

            "mAP75":
                self.map75,

            "mAP95":
                self.map95,

            "mAP50_95":
                self.map,

            # Detection metrics
            "Precision":
                precision,

            "Recall":
                recall,

            "F1-score":
                f1,

            # Confusion counts
            "TP":
                tp,

            "FP":
                fp,

            "FN":
                fn,

            # Per-class AP
            "AP_per_class":
                self.class_AP,

            "AP75_per_class":
                self.class_AP75,

            "AP95_per_class":
                self.class_AP95,

            "AP50_95_per_class":
                self.class_AP_coco,

            # Per-class detection
            "GT_per_class":
                self.class_gt_count,

            "Pred_per_class":
                self.class_pred_count,

            "TP_per_class":
                self.class_tp,

            "FP_per_class":
                self.class_fp,

            "FN_per_class":
                self.class_fn,

            # Configuration
            "IoU_threshold":
                self.iou_threshold,

            "Score_threshold":
                self.score_threshold,

            "Num_images":
                self.num_images,

            "Ground_truth_boxes":
                self.total_gt_boxes,

            "Predicted_boxes":
                self.total_pred_boxes,

        }


# ======================================================================
# TEST
# ======================================================================

if __name__ == "__main__":

    print()

    print(
        "=" * 78
    )

    print(
        "             TESTING SICDA EVALUATOR"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------------
    # Create evaluator
    # --------------------------------------------------------------

    evaluator = Evaluator(
        num_classes=4,
        iou_threshold=0.50,
        score_threshold=0.05,
        model=None,
        device="cuda"
    )

    # --------------------------------------------------------------
    # Dummy target
    # --------------------------------------------------------------

    targets = [

        {
            "boxes": torch.tensor(
                [
                    [10, 10, 50, 50],
                    [100, 100, 150, 160]
                ],
                dtype=torch.float32
            ),

            "labels": torch.tensor(
                [1, 2],
                dtype=torch.long
            )
        }

    ]

    # --------------------------------------------------------------
    # Dummy prediction
    # --------------------------------------------------------------

    detections = [

        {
            "boxes": torch.tensor(
                [
                    [10, 10, 50, 50],
                    [105, 105, 150, 160],
                    [200, 200, 250, 250]
                ],
                dtype=torch.float32
            ),

            "labels": torch.tensor(
                [1, 2, 3],
                dtype=torch.long
            ),

            "scores": torch.tensor(
                [
                    0.95,
                    0.90,
                    0.20
                ],
                dtype=torch.float32
            )
        }

    ]

    # --------------------------------------------------------------
    # Update
    # --------------------------------------------------------------

    evaluator.update(
        detections,
        {
            "targets": targets
        }
    )

    # --------------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------------

    results = evaluator.evaluate()

    # --------------------------------------------------------------
    # Test results
    # --------------------------------------------------------------

    assert results["mAP50"] >= 0.0

    assert results["mAP50"] <= 1.0

    assert results["mAP75"] >= 0.0

    assert results["mAP75"] <= 1.0

    assert results["mAP95"] >= 0.0

    assert results["mAP95"] <= 1.0

    assert results["mAP50_95"] >= 0.0

    assert results["mAP50_95"] <= 1.0

    assert results["Precision"] >= 0.0

    assert results["Recall"] >= 0.0

    assert results["F1-score"] >= 0.0

    print()

    print(
        "=" * 78
    )

    print(
        "SICDA EVALUATOR TEST PASSED"
    )

    print(
        "=" * 78
    )