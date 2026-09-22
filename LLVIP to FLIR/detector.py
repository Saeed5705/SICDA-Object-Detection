"""
=======================================================================
SICDA Detector
=======================================================================

Task:
    LLVIP -> FLIR ADAS v2

Detector:
    Faster R-CNN

Input:
    SICDA fused multi-scale features

        p2
        p3
        p4
        p5

Output:
    Training:
        Faster R-CNN loss dictionary

    Inference:
        list of detection dictionaries

IMPORTANT DESIGN:
    SICDA computes FPN features BEFORE Faster R-CNN.

    Therefore Faster R-CNN MUST NOT resize the images internally.

    This implementation uses a custom NoResizeRCNNTransform that:

        1. Does NOT resize images
        2. Does NOT modify target coordinates
        3. Performs normalization only
        4. Performs standard batch padding

=======================================================================
"""

import os
from collections import OrderedDict

import torch
import torch.nn as nn

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.transform import (
    GeneralizedRCNNTransform
)

from config import Config


# =====================================================================
# CONFIGURATION HELPERS
# =====================================================================

def _get_config_value(
    name,
    default
):
    """
    Safely read a configuration value.

    If the attribute does not exist in Config, the supplied default
    value is returned.

    This prevents detector.py from crashing when optional detector
    parameters are not present in config.py.
    """

    return getattr(
        Config,
        name,
        default
    )


# =====================================================================
# DEFAULT DETECTOR CONFIGURATION
# =====================================================================

# ---------------------------------------------------------------------
# FPN levels
# ---------------------------------------------------------------------

DEFAULT_ANCHOR_SIZES = (

    (32,),

    (64,),

    (128,),

    (256,)

)


DEFAULT_ANCHOR_ASPECT_RATIOS = (

    (0.5, 1.0, 2.0),

    (0.5, 1.0, 2.0),

    (0.5, 1.0, 2.0),

    (0.5, 1.0, 2.0)

)


# ---------------------------------------------------------------------
# RPN defaults
#
# These are torchvision Faster R-CNN style defaults.
# ---------------------------------------------------------------------

DEFAULT_RPN_PRE_NMS_TOP_N_TRAIN = 2000

DEFAULT_RPN_POST_NMS_TOP_N_TRAIN = 1000

DEFAULT_RPN_PRE_NMS_TOP_N_TEST = 1000

DEFAULT_RPN_POST_NMS_TOP_N_TEST = 1000

DEFAULT_RPN_BATCH_SIZE_PER_IMAGE = 256


# ---------------------------------------------------------------------
# ROI defaults
# ---------------------------------------------------------------------

DEFAULT_BOX_BATCH_SIZE_PER_IMAGE = 512

DEFAULT_BOX_DETECTIONS_PER_IMG = 100


# ---------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------

DEFAULT_SCORE_THRESHOLD = 0.05

DEFAULT_NMS_THRESHOLD = 0.50


# ---------------------------------------------------------------------
# ROI pool
# ---------------------------------------------------------------------

DEFAULT_ROI_OUTPUT_SIZE = 7

DEFAULT_ROI_SAMPLING_RATIO = 2


# =====================================================================
# RESOLVE DETECTOR CONFIGURATION
# =====================================================================

def resolve_detector_config():
    """
    Resolve all detector-specific configuration values.

    Config.py does NOT need to contain every detector parameter.

    Missing parameters automatically receive safe defaults.
    """

    anchor_sizes = _get_config_value(
        "ANCHOR_SIZES",
        DEFAULT_ANCHOR_SIZES
    )

    anchor_ratios = _get_config_value(
        "ANCHOR_ASPECT_RATIOS",
        DEFAULT_ANCHOR_ASPECT_RATIOS
    )

    rpn_pre_nms_train = int(
        _get_config_value(
            "RPN_PRE_NMS_TOP_N_TRAIN",
            DEFAULT_RPN_PRE_NMS_TOP_N_TRAIN
        )
    )

    rpn_post_nms_train = int(
        _get_config_value(
            "RPN_POST_NMS_TOP_N_TRAIN",
            DEFAULT_RPN_POST_NMS_TOP_N_TRAIN
        )
    )

    rpn_pre_nms_test = int(
        _get_config_value(
            "RPN_PRE_NMS_TOP_N_TEST",
            DEFAULT_RPN_PRE_NMS_TOP_N_TEST
        )
    )

    rpn_post_nms_test = int(
        _get_config_value(
            "RPN_POST_NMS_TOP_N_TEST",
            DEFAULT_RPN_POST_NMS_TOP_N_TEST
        )
    )

    rpn_batch_size = int(
        _get_config_value(
            "RPN_BATCH_SIZE_PER_IMAGE",
            DEFAULT_RPN_BATCH_SIZE_PER_IMAGE
        )
    )

    box_batch_size = int(
        _get_config_value(
            "BOX_BATCH_SIZE_PER_IMAGE",
            DEFAULT_BOX_BATCH_SIZE_PER_IMAGE
        )
    )

    box_detections = int(
        _get_config_value(
            "BOX_DETECTIONS_PER_IMG",
            DEFAULT_BOX_DETECTIONS_PER_IMG
        )
    )

    score_threshold = float(
        _get_config_value(
            "SCORE_THRESHOLD",
            DEFAULT_SCORE_THRESHOLD
        )
    )

    nms_threshold = float(
        _get_config_value(
            "NMS_THRESHOLD",
            DEFAULT_NMS_THRESHOLD
        )
    )

    roi_output_size = int(
        _get_config_value(
            "ROI_OUTPUT_SIZE",
            DEFAULT_ROI_OUTPUT_SIZE
        )
    )

    roi_sampling_ratio = int(
        _get_config_value(
            "ROI_SAMPLING_RATIO",
            DEFAULT_ROI_SAMPLING_RATIO
        )
    )

    # -----------------------------------------------------------------
    # Validate anchors
    # -----------------------------------------------------------------

    if len(anchor_sizes) != 4:

        raise ValueError(
            "ANCHOR_SIZES must contain exactly "
            "4 levels for p2-p5."
        )

    if len(anchor_ratios) != 4:

        raise ValueError(
            "ANCHOR_ASPECT_RATIOS must contain exactly "
            "4 levels for p2-p5."
        )

    return {

        "anchor_sizes":
            tuple(
                tuple(x)
                for x in anchor_sizes
            ),

        "anchor_ratios":
            tuple(
                tuple(x)
                for x in anchor_ratios
            ),

        "rpn_pre_nms_train":
            rpn_pre_nms_train,

        "rpn_post_nms_train":
            rpn_post_nms_train,

        "rpn_pre_nms_test":
            rpn_pre_nms_test,

        "rpn_post_nms_test":
            rpn_post_nms_test,

        "rpn_batch_size":
            rpn_batch_size,

        "box_batch_size":
            box_batch_size,

        "box_detections":
            box_detections,

        "score_threshold":
            score_threshold,

        "nms_threshold":
            nms_threshold,

        "roi_output_size":
            roi_output_size,

        "roi_sampling_ratio":
            roi_sampling_ratio

    }


# =====================================================================
# PRINT RESOLVED CONFIGURATION
# =====================================================================

def print_anchor_configuration():

    cfg = resolve_detector_config()

    print()

    print("=" * 70)

    print(
        "ANCHOR CONFIGURATION"
    )

    print("=" * 70)

    has_anchor_config = (
        hasattr(
            Config,
            "ANCHOR_SIZES"
        )
    )

    if has_anchor_config:

        print(
            "Source: Config.ANCHOR_SIZES"
        )

    else:

        print(
            "Source: detector.py defaults"
        )

    print()

    print(
        "Anchor Sizes:"
    )

    for index, size in enumerate(
        cfg["anchor_sizes"]
    ):

        print(
            f"  P{index + 2}: {size}"
        )

    print()

    print(
        "Anchor Ratios:"
    )

    for index, ratios in enumerate(
        cfg["anchor_ratios"]
    ):

        print(
            f"  P{index + 2}: {ratios}"
        )

    print("=" * 70)


# =====================================================================
# TRUE NO-RESIZE TRANSFORM
# =====================================================================

class NoResizeRCNNTransform(
    GeneralizedRCNNTransform
):
    """
    Custom Faster R-CNN transform.

    torchvision normally resizes images according to min_size/max_size.

    SICDA has already generated FPN features according to the original
    image geometry.

    Therefore resizing must be disabled.
    """

    def __init__(
        self,
        image_mean,
        image_std,
        size_divisible=32
    ):

        super().__init__(

            min_size=1,

            max_size=100000,

            image_mean=image_mean,

            image_std=image_std,

            size_divisible=size_divisible

        )

    # -----------------------------------------------------------------
    # DISABLE RESIZE
    # -----------------------------------------------------------------

    def resize(
        self,
        image,
        target
    ):

        return image, target


# =====================================================================
# BACKBONE WRAPPER
# =====================================================================

class BackboneWithFPN(
    nn.Module
):
    """
    Adapter between SICDA fused FPN features and Faster R-CNN.

    SICDA provides:

        p2
        p3
        p4
        p5

    Faster R-CNN receives:

        "0" -> p2
        "1" -> p3
        "2" -> p4
        "3" -> p5
    """

    def __init__(self):

        super().__init__()

        self.out_channels = int(
            _get_config_value(
                "FEATURE_DIM",
                256
            )
        )

        self.features = None

        self.feature_shapes = {}

    # -----------------------------------------------------------------
    # SET FEATURES
    # -----------------------------------------------------------------

    def set_features(
        self,
        fused_features
    ):

        if not isinstance(
            fused_features,
            dict
        ):

            raise TypeError(
                "fused_features must be a dictionary."
            )

        required_levels = (

            "p2",

            "p3",

            "p4",

            "p5"

        )

        missing = [

            level

            for level in required_levels

            if level not in fused_features

        ]

        if missing:

            raise KeyError(

                "Missing SICDA FPN levels: "

                f"{missing}. "

                f"Available: "

                f"{list(fused_features.keys())}"

            )

        converted = OrderedDict()

        batch_size = None

        for index, level in enumerate(
            required_levels
        ):

            feature = fused_features[
                level
            ]

            if not torch.is_tensor(
                feature
            ):

                raise TypeError(

                    f"Feature {level} must be Tensor, "

                    f"got {type(feature)}"

                )

            if feature.ndim != 4:

                raise ValueError(

                    f"Feature {level} must have shape "

                    f"[B,C,H,W]. "

                    f"Got {tuple(feature.shape)}"

                )

            if feature.shape[1] != self.out_channels:

                raise ValueError(

                    f"Feature {level} has "

                    f"{feature.shape[1]} channels, "

                    f"expected "

                    f"{self.out_channels}."

                )

            if not torch.isfinite(
                feature
            ).all():

                raise RuntimeError(

                    f"Feature {level} contains "

                    "NaN or Inf."

                )

            if batch_size is None:

                batch_size = feature.shape[0]

            elif feature.shape[0] != batch_size:

                raise ValueError(

                    f"FPN batch mismatch at {level}: "

                    f"{feature.shape[0]} != "

                    f"{batch_size}"

                )

            converted[
                str(index)
            ] = feature

            self.feature_shapes[
                level
            ] = tuple(
                feature.shape
            )

        self.features = converted

    # -----------------------------------------------------------------
    # FORWARD
    # -----------------------------------------------------------------

    def forward(
        self,
        images
    ):

        if self.features is None:

            raise RuntimeError(

                "SICDA FPN features are not set. "

                "Call set_features() before detector forward."

            )

        return self.features

    # -----------------------------------------------------------------
    # CLEAR
    # -----------------------------------------------------------------

    def clear_features(
        self
    ):

        self.features = None

        self.feature_shapes = {}


# =====================================================================
# SICDA DETECTOR
# =====================================================================

class SICDADetector(
    nn.Module
):
    """
    SICDA Faster R-CNN detector.

    Pipeline:

        RGB + Thermal
              |
              v
        Dual Backbone
              |
              v
        Cross Modal Attention
              |
              v
        Illumination Fusion
              |
              v
        SICDA FPN
              |
              v
        p2 / p3 / p4 / p5
              |
              v
        Faster R-CNN
              |
              v
        Detection
    """

    def __init__(
        self
    ):

        super().__init__()

        # =============================================================
        # RESOLVE CONFIGURATION
        # =============================================================

        detector_cfg = (
            resolve_detector_config()
        )

        # Store detector configuration locally.

        self.anchor_sizes = (
            detector_cfg[
                "anchor_sizes"
            ]
        )

        self.anchor_aspect_ratios = (
            detector_cfg[
                "anchor_ratios"
            ]
        )

        self.rpn_pre_nms_train = (
            detector_cfg[
                "rpn_pre_nms_train"
            ]
        )

        self.rpn_post_nms_train = (
            detector_cfg[
                "rpn_post_nms_train"
            ]
        )

        self.rpn_pre_nms_test = (
            detector_cfg[
                "rpn_pre_nms_test"
            ]
        )

        self.rpn_post_nms_test = (
            detector_cfg[
                "rpn_post_nms_test"
            ]
        )

        self.rpn_batch_size = (
            detector_cfg[
                "rpn_batch_size"
            ]
        )

        self.box_batch_size = (
            detector_cfg[
                "box_batch_size"
            ]
        )

        self.box_detections = (
            detector_cfg[
                "box_detections"
            ]
        )

        self.score_threshold = (
            detector_cfg[
                "score_threshold"
            ]
        )

        self.nms_threshold = (
            detector_cfg[
                "nms_threshold"
            ]
        )

        self.roi_output_size = (
            detector_cfg[
                "roi_output_size"
            ]
        )

        self.roi_sampling_ratio = (
            detector_cfg[
                "roi_sampling_ratio"
            ]
        )

        # =============================================================
        # BASIC CONFIGURATION
        # =============================================================

        self.num_classes = int(
            _get_config_value(
                "NUM_CLASSES",
                4
            )
        )

        self.feature_dim = int(
            _get_config_value(
                "FEATURE_DIM",
                256
            )
        )

        # =============================================================
        # BACKBONE ADAPTER
        # =============================================================

        self.fpn_wrapper = (
            BackboneWithFPN()
        )

        # Make sure wrapper channels match Config.FEATURE_DIM.

        if (
            self.fpn_wrapper.out_channels
            !=
            self.feature_dim
        ):

            raise ValueError(

                "Backbone FEATURE_DIM mismatch: "

                f"wrapper={self.fpn_wrapper.out_channels}, "

                f"config={self.feature_dim}"

            )

        # =============================================================
        # ANCHOR GENERATOR
        # =============================================================

        anchor_generator = AnchorGenerator(

            sizes=self.anchor_sizes,

            aspect_ratios=self.anchor_aspect_ratios

        )

        # =============================================================
        # ROI POOLER
        # =============================================================

        roi_pooler = MultiScaleRoIAlign(

            featmap_names=[

                "0",

                "1",

                "2",

                "3"

            ],

            output_size=(
                self.roi_output_size
            ),

            sampling_ratio=(
                self.roi_sampling_ratio
            )

        )

        # =============================================================
        # FASTER R-CNN
        # =============================================================

        self.detector = FasterRCNN(

            backbone=self.fpn_wrapper,

            num_classes=self.num_classes,

            rpn_anchor_generator=(
                anchor_generator
            ),

            box_roi_pool=(
                roi_pooler
            ),

            # ---------------------------------------------------------
            # Dummy values.
            #
            # Replaced with NoResizeRCNNTransform below.
            # ---------------------------------------------------------

            min_size=1,

            max_size=100000,

            # ---------------------------------------------------------
            # RPN
            # ---------------------------------------------------------

            rpn_pre_nms_top_n_train=(
                self.rpn_pre_nms_train
            ),

            rpn_post_nms_top_n_train=(
                self.rpn_post_nms_train
            ),

            rpn_pre_nms_top_n_test=(
                self.rpn_pre_nms_test
            ),

            rpn_post_nms_top_n_test=(
                self.rpn_post_nms_test
            ),

            rpn_batch_size_per_image=(
                self.rpn_batch_size
            ),

            # ---------------------------------------------------------
            # ROI
            # ---------------------------------------------------------

            box_batch_size_per_image=(
                self.box_batch_size
            ),

            box_detections_per_img=(
                self.box_detections
            ),

            # ---------------------------------------------------------
            # Detection thresholds
            # ---------------------------------------------------------

            box_score_thresh=(
                self.score_threshold
            ),

            box_nms_thresh=(
                self.nms_threshold
            )

        )

        # =============================================================
        # NO-RESIZE TRANSFORM
        # =============================================================

        self._configure_no_resize_transform()

    # =================================================================
    # NO-RESIZE TRANSFORM
    # =================================================================

    def _configure_no_resize_transform(
        self
    ):

        self.detector.transform = (
            NoResizeRCNNTransform(

                image_mean=[

                    0.0,

                    0.0,

                    0.0

                ],

                image_std=[

                    1.0,

                    1.0,

                    1.0

                ],

                size_divisible=32

            )
        )

    # =================================================================
    # IMAGE VALIDATION
    # =================================================================

    @staticmethod
    def _validate_images(
        images
    ):

        if not isinstance(
            images,
            (list, tuple)
        ):

            raise TypeError(
                "images must be list or tuple."
            )

        if len(images) == 0:

            raise ValueError(
                "Received empty image list."
            )

        for index, image in enumerate(
            images
        ):

            if not torch.is_tensor(
                image
            ):

                raise TypeError(

                    f"Image {index} is not a Tensor."

                )

            if image.ndim != 3:

                raise ValueError(

                    f"Image {index} must have "

                    f"[C,H,W], got "

                    f"{tuple(image.shape)}"

                )

            if image.shape[0] != 3:

                raise ValueError(

                    f"Image {index} must have 3 channels, "

                    f"got {image.shape[0]}."

                )

            if (
                image.shape[-2] <= 0
                or
                image.shape[-1] <= 0
            ):

                raise ValueError(

                    f"Image {index} has invalid spatial size "

                    f"{tuple(image.shape[-2:])}"

                )

            if not torch.isfinite(
                image
            ).all():

                raise RuntimeError(

                    f"Image {index} contains NaN/Inf."

                )

    # =================================================================
    # FEATURE VALIDATION
    # =================================================================

    def validate_features(
        self,
        fused_features,
        images
    ):

        required = (

            "p2",

            "p3",

            "p4",

            "p5"

        )

        if not isinstance(
            fused_features,
            dict
        ):

            raise TypeError(
                "fused_features must be dict."
            )

        self._validate_images(
            images
        )

        batch_size = len(
            images
        )

        for level in required:

            if level not in fused_features:

                raise KeyError(

                    f"Missing feature level: {level}"

                )

            feature = fused_features[
                level
            ]

            if not torch.is_tensor(
                feature
            ):

                raise TypeError(

                    f"{level} is not a Tensor."

                )

            if feature.ndim != 4:

                raise ValueError(

                    f"{level}: expected [B,C,H,W], "

                    f"got {tuple(feature.shape)}"

                )

            if feature.shape[0] != batch_size:

                raise ValueError(

                    f"{level}: feature batch size "

                    f"{feature.shape[0]} != "

                    f"image batch size {batch_size}"

                )

            if feature.shape[1] != self.feature_dim:

                raise ValueError(

                    f"{level}: channels "

                    f"{feature.shape[1]} != "

                    f"FEATURE_DIM "

                    f"{self.feature_dim}"

                )

            if (
                feature.shape[-2] < 1
                or
                feature.shape[-1] < 1
            ):

                raise ValueError(

                    f"{level}: invalid spatial dimensions."

                )

            if not torch.isfinite(
                feature
            ).all():

                raise RuntimeError(

                    f"{level} contains NaN/Inf."

                )

    # =================================================================
    # FPN STRIDE REPORT
    # =================================================================

    @staticmethod
    def feature_stride_report(
        fused_features,
        image_height,
        image_width
    ):

        report = {}

        for level in (

            "p2",

            "p3",

            "p4",

            "p5"

        ):

            feature = fused_features[
                level
            ]

            h = feature.shape[-2]

            w = feature.shape[-1]

            stride_h = (
                float(image_height)
                /
                float(h)
            )

            stride_w = (
                float(image_width)
                /
                float(w)
            )

            report[level] = (

                stride_h,

                stride_w

            )

        return report

    # =================================================================
    # FEATURE ALIGNMENT
    # =================================================================

    def check_feature_alignment(
        self,
        fused_features,
        images,
        tolerance=0.25
    ):

        expected = {

            "p2": 4.0,

            "p3": 8.0,

            "p4": 16.0,

            "p5": 32.0

        }

        if torch.is_tensor(
            images
        ):

            height = images.shape[-2]

            width = images.shape[-1]

        else:

            height = images[0].shape[-2]

            width = images[0].shape[-1]

        report = (
            self.feature_stride_report(

                fused_features,

                height,

                width

            )
        )

        results = {}

        for level in expected:

            stride_h, stride_w = (
                report[level]
            )

            expected_stride = (
                expected[level]
            )

            error_h = (

                abs(
                    stride_h
                    -
                    expected_stride
                )
                /
                expected_stride

            )

            error_w = (

                abs(
                    stride_w
                    -
                    expected_stride
                )
                /
                expected_stride

            )

            results[level] = {

                "stride_h":
                    stride_h,

                "stride_w":
                    stride_w,

                "expected":
                    expected_stride,

                "aligned":
                    (
                        error_h <= tolerance
                        and
                        error_w <= tolerance
                    )

            }

        return results

    # =================================================================
    # PRINT ALIGNMENT REPORT
    # =================================================================

    def print_alignment_report(
        self,
        fused_features,
        images
    ):

        if torch.is_tensor(
            images
        ):

            height = images.shape[-2]

            width = images.shape[-1]

        else:

            height = images[0].shape[-2]

            width = images[0].shape[-1]

        report = (
            self.feature_stride_report(

                fused_features,

                height,

                width

            )
        )

        print()

        print("=" * 78)

        print(
            "SICDA FPN / IMAGE ALIGNMENT"
        )

        print("=" * 78)

        print(
            f"Input image: "
            f"{height} x {width}"
        )

        expected = {

            "p2": 4,

            "p3": 8,

            "p4": 16,

            "p5": 32

        }

        for level in (

            "p2",

            "p3",

            "p4",

            "p5"

        ):

            feature = fused_features[
                level
            ]

            h = feature.shape[-2]

            w = feature.shape[-1]

            sh, sw = report[level]

            print(

                f"{level}: "

                f"{h} x {w} | "

                f"stride=({sh:.2f},{sw:.2f}) | "

                f"expected={expected[level]}"

            )

        print("=" * 78)

        return report

    # =================================================================
    # FORWARD
    # =================================================================

    def forward(
        self,
        fused_features,
        images,
        targets=None
    ):

        # -------------------------------------------------------------
        # Convert [B,C,H,W] to list[C,H,W]
        # -------------------------------------------------------------

        if torch.is_tensor(
            images
        ):

            if images.ndim != 4:

                raise ValueError(

                    "Tensor images must have shape "

                    "[B,C,H,W]."

                )

            image_list = [

                images[i]

                for i in range(
                    images.shape[0]
                )

            ]

        else:

            image_list = list(
                images
            )

        # -------------------------------------------------------------
        # Validate
        # -------------------------------------------------------------

        self.validate_features(

            fused_features,

            image_list

        )

        # -------------------------------------------------------------
        # Device check
        # -------------------------------------------------------------

        image_device = (
            image_list[0].device
        )

        for level in (

            "p2",

            "p3",

            "p4",

            "p5"

        ):

            if (
                fused_features[level].device
                !=
                image_device
            ):

                raise RuntimeError(

                    f"{level} is on "

                    f"{fused_features[level].device}, "

                    f"but images are on "

                    f"{image_device}."

                )

        # -------------------------------------------------------------
        # Set SICDA features
        # -------------------------------------------------------------

        self.fpn_wrapper.set_features(
            fused_features
        )

        try:

            # =========================================================
            # TRAINING
            # =========================================================

            if self.training:

                if targets is None:

                    raise ValueError(

                        "Targets are required "
                        "during training."

                    )

                if len(targets) != len(
                    image_list
                ):

                    raise ValueError(

                        f"Targets={len(targets)} but "

                        f"images={len(image_list)}."

                    )

                targets = (
                    self._validate_targets(

                        targets,

                        image_list

                    )
                )

                losses = self.detector(

                    image_list,

                    targets

                )

                return losses

            # =========================================================
            # INFERENCE
            # =========================================================

            detections = self.detector(
                image_list
            )

            return detections

        finally:

            self.fpn_wrapper.clear_features()

    # =================================================================
    # TARGET VALIDATION
    # =================================================================

    def _validate_targets(
        self,
        targets,
        images
    ):

        validated = []

        for index, target in enumerate(
            targets
        ):

            if not isinstance(
                target,
                dict
            ):

                raise TypeError(

                    f"Target {index} must be dict."

                )

            if "boxes" not in target:

                raise KeyError(

                    f"Target {index} has no 'boxes'."

                )

            if "labels" not in target:

                raise KeyError(

                    f"Target {index} has no 'labels'."

                )

            # ---------------------------------------------------------
            # BOXES
            # ---------------------------------------------------------

            boxes = target[
                "boxes"
            ]

            if not torch.is_tensor(
                boxes
            ):

                boxes = torch.as_tensor(

                    boxes,

                    dtype=torch.float32

                )

            boxes = boxes.to(

                device=images[index].device,

                dtype=torch.float32

            )

            if boxes.numel() == 0:

                boxes = boxes.reshape(
                    0,
                    4
                )

            if boxes.ndim != 2:

                raise ValueError(

                    f"Target {index} boxes must be "

                    f"[N,4]. Got {tuple(boxes.shape)}"

                )

            if boxes.shape[-1] != 4:

                raise ValueError(

                    f"Target {index} boxes must "

                    "have exactly 4 coordinates."

                )

            # ---------------------------------------------------------
            # LABELS
            # ---------------------------------------------------------

            labels = target[
                "labels"
            ]

            if not torch.is_tensor(
                labels
            ):

                labels = torch.as_tensor(

                    labels,

                    dtype=torch.int64

                )

            labels = labels.to(

                device=images[index].device,

                dtype=torch.int64

            )

            labels = labels.reshape(
                -1
            )

            if len(labels) != len(
                boxes
            ):

                raise ValueError(

                    f"Target {index}: "

                    f"{len(boxes)} boxes but "

                    f"{len(labels)} labels."

                )

            # ---------------------------------------------------------
            # LABEL RANGE
            # ---------------------------------------------------------

            if len(labels) > 0:

                invalid_labels = (

                    (labels < 1)

                    |

                    (
                        labels
                        >=
                        self.num_classes
                    )

                )

                if invalid_labels.any():

                    bad = labels[
                        invalid_labels
                    ].detach().cpu().tolist()

                    raise ValueError(

                        f"Target {index} contains invalid "

                        f"class IDs {bad}. "

                        f"Valid foreground IDs are "

                        f"1..{self.num_classes - 1}."

                    )

            # ---------------------------------------------------------
            # FINITE BOX CHECK
            # ---------------------------------------------------------

            if not torch.isfinite(
                boxes
            ).all():

                raise ValueError(

                    f"Target {index} contains NaN/Inf boxes."

                )

            # ---------------------------------------------------------
            # IMAGE DIMENSIONS
            # ---------------------------------------------------------

            height = int(
                images[index].shape[-2]
            )

            width = int(
                images[index].shape[-1]
            )

            # ---------------------------------------------------------
            # VALID BOX FILTER
            # ---------------------------------------------------------

            if len(boxes) > 0:

                x1 = boxes[:, 0]

                y1 = boxes[:, 1]

                x2 = boxes[:, 2]

                y2 = boxes[:, 3]

                valid = (

                    (x2 > x1)

                    &

                    (y2 > y1)

                    &

                    (x1 < width)

                    &

                    (y1 < height)

                    &

                    (x2 > 0)

                    &

                    (y2 > 0)

                )

                boxes = boxes[
                    valid
                ]

                labels = labels[
                    valid
                ]

            # ---------------------------------------------------------
            # CLAMP
            # ---------------------------------------------------------

            if len(boxes) > 0:

                boxes[:, 0].clamp_(
                    0,
                    max(
                        width - 1,
                        0
                    )
                )

                boxes[:, 2].clamp_(
                    0,
                    max(
                        width - 1,
                        0
                    )
                )

                boxes[:, 1].clamp_(
                    0,
                    max(
                        height - 1,
                        0
                    )
                )

                boxes[:, 3].clamp_(
                    0,
                    max(
                        height - 1,
                        0
                    )
                )

            # ---------------------------------------------------------
            # FINAL VALIDITY CHECK
            # ---------------------------------------------------------

            if len(boxes) > 0:

                valid_after_clamp = (

                    (
                        boxes[:, 2]
                        >
                        boxes[:, 0]
                    )

                    &

                    (
                        boxes[:, 3]
                        >
                        boxes[:, 1]
                    )

                )

                boxes = boxes[
                    valid_after_clamp
                ]

                labels = labels[
                    valid_after_clamp
                ]

            # ---------------------------------------------------------
            # OUTPUT
            # ---------------------------------------------------------

            new_target = dict(
                target
            )

            new_target[
                "boxes"
            ] = boxes

            new_target[
                "labels"
            ] = labels

            validated.append(
                new_target
            )

        return validated

    # =================================================================
    # TOTAL DETECTION LOSS
    # =================================================================

    @staticmethod
    def total_detection_loss(
        loss_dict
    ):

        if not isinstance(
            loss_dict,
            dict
        ):

            raise TypeError(
                "loss_dict must be dict."
            )

        total_loss = None

        for name, loss in (
            loss_dict.items()
        ):

            if not torch.is_tensor(
                loss
            ):

                raise TypeError(

                    f"Loss '{name}' must be Tensor."

                )

            if not torch.isfinite(
                loss
            ).all():

                raise RuntimeError(

                    f"Loss '{name}' contains NaN/Inf."

                )

            if total_loss is None:

                total_loss = loss

            else:

                total_loss = (

                    total_loss
                    +
                    loss

                )

        if total_loss is None:

            return torch.tensor(
                0.0
            )

        return total_loss

    # =================================================================
    # PREDICT
    # =================================================================

    @torch.no_grad()
    def predict(
        self,
        fused_features,
        images
    ):

        previous_mode = self.training

        self.eval()

        try:

            detections = self.forward(

                fused_features=(
                    fused_features
                ),

                images=images,

                targets=None

            )

        finally:

            if previous_mode:

                self.train()

        return detections

    # =================================================================
    # CLEAR CACHE
    # =================================================================

    def clear_feature_cache(
        self
    ):

        self.fpn_wrapper.clear_features()

    # =================================================================
    # ABLATION STATE
    # =================================================================

    def ablation_state(
        self
    ):

        return {

            "detector":
                "FasterRCNN",

            "num_classes":
                self.num_classes,

            "feature_dim":
                self.feature_dim,

            "fpn_levels":
                [
                    "p2",
                    "p3",
                    "p4",
                    "p5"
                ],

            "anchor_sizes":
                self.anchor_sizes,

            "anchor_aspect_ratios":
                self.anchor_aspect_ratios,

            "rpn_pre_nms_train":
                self.rpn_pre_nms_train,

            "rpn_post_nms_train":
                self.rpn_post_nms_train,

            "rpn_pre_nms_test":
                self.rpn_pre_nms_test,

            "rpn_post_nms_test":
                self.rpn_post_nms_test,

            "rpn_batch_size":
                self.rpn_batch_size,

            "box_batch_size":
                self.box_batch_size,

            "box_detections_per_img":
                self.box_detections,

            "score_threshold":
                self.score_threshold,

            "nms_threshold":
                self.nms_threshold,

            "internal_resize":
                False,

            "transform":
                "NoResizeRCNNTransform"

        }


# =====================================================================
# FACTORY
# =====================================================================

def build_detector():

    return SICDADetector()


# =====================================================================
# DETECTOR INFORMATION
# =====================================================================

def detector_info(
    detector=None
):

    if detector is None:

        detector = build_detector()

    print()

    print("=" * 78)

    print(
        "                         SICDA DETECTOR"
    )

    print("=" * 78)

    print(
        "Detector        : Faster R-CNN"
    )

    print(
        "Backbone Input  : SICDA Fused Features"
    )

    print(
        "Feature Levels  : p2, p3, p4, p5"
    )

    print(
        "Number Classes  :",
        detector.num_classes
    )

    print(
        "Class Names     :",
        _get_config_value(
            "CLASS_NAMES",
            {
                0: "background",
                1: "person",
                2: "car",
                3: "bike"
            }
        )
    )

    print(
        "Feature Dim     :",
        detector.feature_dim
    )

    print(
        "Internal Resize : DISABLED"
    )

    print(
        "Transform       : NoResizeRCNNTransform"
    )

    print()

    print(
        "Anchor Sizes:"
    )

    for i, size in enumerate(
        detector.anchor_sizes
    ):

        print(
            f"  P{i + 2}: {size}"
        )

    print()

    print(
        "Anchor Ratios:"
    )

    for i, ratios in enumerate(
        detector.anchor_aspect_ratios
    ):

        print(
            f"  P{i + 2}: {ratios}"
        )

    print()

    print(
        "RPN Train Pre-NMS  :",
        detector.rpn_pre_nms_train
    )

    print(
        "RPN Train Post-NMS :",
        detector.rpn_post_nms_train
    )

    print(
        "RPN Test Pre-NMS   :",
        detector.rpn_pre_nms_test
    )

    print(
        "RPN Test Post-NMS  :",
        detector.rpn_post_nms_test
    )

    print()

    print(
        "ROI Batch/Image     :",
        detector.box_batch_size
    )

    print(
        "RPN Batch/Image     :",
        detector.rpn_batch_size
    )

    print(
        "Max Detections      :",
        detector.box_detections
    )

    print()

    print(
        "NMS Threshold       :",
        detector.nms_threshold
    )

    print(
        "Score Threshold     :",
        detector.score_threshold
    )

    print("=" * 78)


# =====================================================================
# PARAMETER INFORMATION
# =====================================================================

def detector_parameter_info(
    detector
):

    total = sum(

        p.numel()

        for p in detector.parameters()

    )

    trainable = sum(

        p.numel()

        for p in detector.parameters()

        if p.requires_grad

    )

    fp32_size_mb = (

        total
        *
        4
        /
        (1024 ** 2)

    )

    print()

    print("=" * 70)

    print(
        "DETECTOR PARAMETER INFORMATION"
    )

    print("=" * 70)

    print(

        f"Total Parameters     : "
        f"{total:,}"

    )

    print(

        f"Trainable Parameters : "
        f"{trainable:,}"

    )

    print(

        f"Model Size FP32      : "
        f"{fp32_size_mb:.2f} MB"

    )

    print(

        f"Parameters           : "
        f"{total / 1e6:.2f} M"

    )

    print("=" * 70)

    return {

        "parameters":
            total,

        "trainable_parameters":
            trainable,

        "model_size_mb":
            fp32_size_mb

    }


# =====================================================================
# MOVE IMAGES TO DEVICE
# =====================================================================

def move_images_to_device(
    images,
    device
):

    return [

        image.to(

            device,

            non_blocking=True

        )

        for image in images

    ]


# =====================================================================
# MOVE TARGETS TO DEVICE
# =====================================================================

def move_targets_to_device(
    targets,
    device
):

    output = []

    for target in targets:

        converted = {}

        if "boxes" not in target:

            raise KeyError(
                "Target missing 'boxes'."
            )

        if "labels" not in target:

            raise KeyError(
                "Target missing 'labels'."
            )

        converted[
            "boxes"
        ] = target[
            "boxes"
        ].to(

            device,

            non_blocking=True

        )

        converted[
            "labels"
        ] = target[
            "labels"
        ].to(

            device,

            non_blocking=True

        )

        for key in (

            "image_id",

            "area",

            "iscrowd"

        ):

            if key in target:

                value = target[
                    key
                ]

                if torch.is_tensor(
                    value
                ):

                    value = value.to(

                        device,

                        non_blocking=True

                    )

                converted[
                    key
                ] = value

        output.append(
            converted
        )

    return output


# =====================================================================
# LLVIP TEST COLLATE
# =====================================================================

def llvip_test_collate_fn(
    batch
):

    rgb_images = [

        item[
            "rgb"
        ]

        for item in batch

    ]

    thermal_images = [

        item[
            "thermal"
        ]

        for item in batch

    ]

    targets = [

        item[
            "target"
        ]

        for item in batch

    ]

    return (

        rgb_images,

        thermal_images,

        targets

    )


# =====================================================================
# TARGET STATISTICS
# =====================================================================

def print_batch_target_statistics(
    targets
):

    total_boxes = 0

    class_counter = {}

    for target in targets:

        labels = target[
            "labels"
        ]

        total_boxes += len(
            labels
        )

        for label in labels.tolist():

            class_counter[
                label
            ] = (

                class_counter.get(
                    label,
                    0
                )
                +
                1

            )

    print()

    print("=" * 70)

    print(
        "LLVIP BATCH ANNOTATION STATISTICS"
    )

    print("=" * 70)

    print(
        "Images      :",
        len(targets)
    )

    print(
        "Total boxes :",
        total_boxes
    )

    class_names = _get_config_value(

        "CLASS_NAMES",

        {
            0: "background",
            1: "person",
            2: "car",
            3: "bike"
        }

    )

    for label_id, count in sorted(
        class_counter.items()
    ):

        if isinstance(
            class_names,
            dict
        ):

            class_name = class_names.get(

                label_id,

                f"class_{label_id}"

            )

        else:

            if (
                0 <= label_id
                <
                len(class_names)
            ):

                class_name = class_names[
                    label_id
                ]

            else:

                class_name = (
                    f"class_{label_id}"
                )

        print(

            f"{label_id} - "
            f"{class_name:<15}: "
            f"{count}"

        )

    print("=" * 70)


# =====================================================================
# SIMPLE DETECTOR TEST
# =====================================================================

def test_detector_shapes():

    print()

    print("=" * 78)

    print(
        "          SICDA DETECTOR CONFIGURATION TEST"
    )

    print("=" * 78)

    cfg = resolve_detector_config()

    print(
        "Feature levels :",
        len(
            cfg[
                "anchor_sizes"
            ]
        )
    )

    print(
        "Anchor levels  :",
        len(
            cfg[
                "anchor_ratios"
            ]
        )
    )

    print(
        "Number classes :",
        _get_config_value(
            "NUM_CLASSES",
            4
        )
    )

    print(
        "Class names    :",
        _get_config_value(
            "CLASS_NAMES",
            {
                0: "background",
                1: "person",
                2: "car",
                3: "bike"
            }
        )
    )

    print()

    print(
        "Resolved Anchor Sizes:"
    )

    for index, size in enumerate(
        cfg[
            "anchor_sizes"
        ]
    ):

        print(
            f"  P{index + 2}: {size}"
        )

    print()

    print(
        "Resolved Anchor Ratios:"
    )

    for index, ratios in enumerate(
        cfg[
            "anchor_ratios"
        ]
    ):

        print(
            f"  P{index + 2}: {ratios}"
        )

    assert len(
        cfg[
            "anchor_sizes"
        ]
    ) == 4

    assert len(
        cfg[
            "anchor_ratios"
        ]
    ) == 4

    assert (

        _get_config_value(
            "NUM_CLASSES",
            4
        )

        ==

        len(
            _get_config_value(
                "CLASS_NAMES",
                {
                    0: "background",
                    1: "person",
                    2: "car",
                    3: "bike"
                }
            )
        )

    )

    detector = build_detector()

    print()

    print(
        "Detector created successfully."
    )

    detector_info(
        detector
    )

    detector_parameter_info(
        detector
    )

    print()

    print(
        "Detector configuration test PASSED."
    )

    print("=" * 78)

    return detector


# =====================================================================
# FPN ALIGNMENT TEST
# =====================================================================

def test_fpn_alignment(
    fused_features,
    images
):

    if torch.is_tensor(
        images
    ):

        height = images.shape[-2]

        width = images.shape[-1]

    else:

        height = images[0].shape[-2]

        width = images[0].shape[-1]

    report = (
        SICDADetector.feature_stride_report(

            fused_features,

            height,

            width

        )
    )

    print()

    print("=" * 70)

    print(
        "SICDA FPN ALIGNMENT REPORT"
    )

    print("=" * 70)

    print(
        f"Input image : "
        f"{height} x {width}"
    )

    expected = {

        "p2": 4,

        "p3": 8,

        "p4": 16,

        "p5": 32

    }

    for level in (

        "p2",

        "p3",

        "p4",

        "p5"

    ):

        feature = fused_features[
            level
        ]

        h = feature.shape[-2]

        w = feature.shape[-1]

        stride_h, stride_w = (
            report[level]
        )

        print(

            f"{level}: "

            f"{h} x {w} | "

            f"stride ≈ "

            f"({stride_h:.2f}, "

            f"{stride_w:.2f}) | "

            f"expected={expected[level]}"

        )

    print("=" * 70)

    return report


# =====================================================================
# MAIN TEST
# =====================================================================

if __name__ == "__main__":

    print()

    print("=" * 78)

    print(
        "                    TESTING SICDA DETECTOR"
    )

    print("=" * 78)

    # =================================================================
    # DEVICE
    # =================================================================

    device = torch.device(

        _get_config_value(
            "DEVICE",
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

    )

    print()

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(

            "GPU:",

            torch.cuda.get_device_name(
                0
            )

        )

        print(

            "CUDA:",

            torch.version.cuda

        )

    # =================================================================
    # PRINT ANCHOR CONFIG
    # =================================================================

    print_anchor_configuration()

    # =================================================================
    # DETECTOR
    # =================================================================

    detector = (
        test_detector_shapes()
    )

    detector = detector.to(
        device
    )

    detector_parameter_info(
        detector
    )

    # =================================================================
    # DUMMY IMAGE
    # =================================================================

    print()

    print("=" * 70)

    print(
        "RUNNING DUMMY FPN TEST"
    )

    print("=" * 70)

    image_size = int(

        _get_config_value(

            "IMAGE_SIZE",

            320

        )

    )

    # -------------------------------------------------------------
    # IMPORTANT
    #
    # If IMAGE_SIZE is a tuple, support it as well.
    # -------------------------------------------------------------

    if isinstance(
        image_size,
        (tuple, list)
    ):

        image_height = int(
            image_size[0]
        )

        image_width = int(
            image_size[1]
        )

    else:

        image_height = int(
            image_size
        )

        image_width = int(
            image_size
        )

    dummy_image = torch.rand(

        3,

        image_height,

        image_width,

        device=device

    )

    dummy_images = [

        dummy_image

    ]

    # =================================================================
    # DUMMY FPN
    # =================================================================

    p2_height = max(

        1,

        image_height // 4

    )

    p2_width = max(

        1,

        image_width // 4

    )

    p3_height = max(

        1,

        image_height // 8

    )

    p3_width = max(

        1,

        image_width // 8

    )

    p4_height = max(

        1,

        image_height // 16

    )

    p4_width = max(

        1,

        image_width // 16

    )

    p5_height = max(

        1,

        image_height // 32

    )

    p5_width = max(

        1,

        image_width // 32

    )

    dummy_features = {

        "p2":
            torch.randn(

                1,

                int(
                    _get_config_value(
                        "FEATURE_DIM",
                        256
                    )
                ),

                p2_height,

                p2_width,

                device=device

            ),

        "p3":
            torch.randn(

                1,

                int(
                    _get_config_value(
                        "FEATURE_DIM",
                        256
                    )
                ),

                p3_height,

                p3_width,

                device=device

            ),

        "p4":
            torch.randn(

                1,

                int(
                    _get_config_value(
                        "FEATURE_DIM",
                        256
                    )
                ),

                p4_height,

                p4_width,

                device=device

            ),

        "p5":
            torch.randn(

                1,

                int(
                    _get_config_value(
                        "FEATURE_DIM",
                        256
                    )
                ),

                p5_height,

                p5_width,

                device=device

            )

    }

    test_fpn_alignment(

        dummy_features,

        dummy_images

    )

    # =================================================================
    # TARGET
    # =================================================================

    dummy_target = {

        "boxes":
            torch.tensor(

                [

                    [

                        20.0,

                        20.0,

                        min(
                            100.0,
                            image_width - 1
                        ),

                        min(
                            120.0,
                            image_height - 1
                        )

                    ]

                ],

                dtype=torch.float32,

                device=device

            ),

        "labels":
            torch.tensor(

                [1],

                dtype=torch.int64,

                device=device

            )

    }

    # =================================================================
    # TRAINING TEST
    # =================================================================

    print()

    print("=" * 70)

    print(
        "FASTER R-CNN TRAINING TEST"
    )

    print("=" * 70)

    detector.train()

    loss_dict = detector(

        fused_features=dummy_features,

        images=dummy_images,

        targets=[

            dummy_target

        ]

    )

    total_loss = (

        detector.total_detection_loss(

            loss_dict

        )

    )

    for name, value in (
        loss_dict.items()
    ):

        print(

            f"{name:<30}: "
            f"{value.detach().item():.6f}"

        )

    print("-" * 70)

    print(

        f"{'Total Detection Loss':<30}: "
        f"{total_loss.detach().item():.6f}"

    )

    if not torch.isfinite(
        total_loss
    ):

        raise RuntimeError(

            "Detection loss is NaN/Inf."

        )

    print()

    print(
        "Training test PASSED."
    )

    # =================================================================
    # CLEANUP
    # =================================================================

    del loss_dict

    del total_loss

    detector.clear_feature_cache()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # =================================================================
    # INFERENCE TEST
    # =================================================================

    print()

    print("=" * 70)

    print(
        "FASTER R-CNN INFERENCE TEST"
    )

    print("=" * 70)

    detector.eval()

    with torch.no_grad():

        detections = detector(

            fused_features=dummy_features,

            images=dummy_images

        )

    print()

    print(

        "Number of detections:",

        len(detections)

    )

    for index, detection in enumerate(
        detections
    ):

        print()

        print(
            f"Image {index}"
        )

        print(

            "Boxes :",

            len(
                detection[
                    "boxes"
                ]
            )

        )

        print(

            "Labels:",

            len(
                detection[
                    "labels"
                ]
            )

        )

        print(

            "Scores:",

            len(
                detection[
                    "scores"
                ]
            )

        )

        if len(
            detection[
                "scores"
            ]
        ) > 0:

            print(

                "Highest score:",

                float(

                    detection[
                        "scores"
                    ].max()

                )

            )

    detector.clear_feature_cache()

    # =================================================================
    # GPU MEMORY
    # =================================================================

    if torch.cuda.is_available():

        print()

        print("=" * 70)

        print(
            "GPU MEMORY"
        )

        print("=" * 70)

        allocated = (

            torch.cuda.memory_allocated(
                device
            )

            /

            (1024 ** 3)

        )

        reserved = (

            torch.cuda.memory_reserved(
                device
            )

            /

            (1024 ** 3)

        )

        peak = (

            torch.cuda.max_memory_allocated(
                device
            )

            /

            (1024 ** 3)

        )

        print(

            f"Allocated : "
            f"{allocated:.3f} GB"

        )

        print(

            f"Reserved  : "
            f"{reserved:.3f} GB"

        )

        print(

            f"Peak      : "
            f"{peak:.3f} GB"

        )

    # =================================================================
    # FINAL
    # =================================================================

    print()

    print("=" * 78)

    print(
        "          DETECTOR TEST COMPLETED SUCCESSFULLY"
    )

    print("=" * 78)

    print()

    print(
        "SICDA detector pipeline:"
    )

    print(
        "  RGB + Thermal"
    )

    print(
        "       ↓"
    )

    print(
        "  Dual Backbone"
    )

    print(
        "       ↓"
    )

    print(
        "  Cross Modal Attention"
    )

    print(
        "       ↓"
    )

    print(
        "  Illumination Fusion"
    )

    print(
        "       ↓"
    )

    print(
        "  p2 / p3 / p4 / p5"
    )

    print(
        "       ↓"
    )

    print(
        "  Faster R-CNN"
    )

    print(
        "       ↓"
    )

    print(
        "  Detection"
    )

    print()

    print(

        "Classes:",

        _get_config_value(

            "CLASS_NAMES",

            {
                0: "background",
                1: "person",
                2: "car",
                3: "bike"
            }

        )

    )

    print("=" * 78)