import torch
import torch.nn as nn

from collections import OrderedDict

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from config import Config


###############################################################
# Backbone Wrapper
###############################################################

class BackboneWithFPN(nn.Module):
    """
    Wrapper used by FasterRCNN.

    FasterRCNN internally calls:

        backbone(images)

    SICDA instead provides already-computed
    fused multi-scale features from:

        p2
        p3
        p4
        p5

    These features are passed directly to Faster R-CNN.
    """

    def __init__(self):

        super().__init__()

        # FasterRCNN requires backbone.out_channels
        self.out_channels = Config.FEATURE_DIM

        self.features = None


    def set_features(self, fused_features):
        """
        Store SICDA fused FPN features.

        Expected keys:

            p2
            p3
            p4
            p5
        """

        required_levels = ["p2", "p3", "p4", "p5"]

        for level in required_levels:

            if level not in fused_features:

                raise KeyError(
                    f"Missing feature level '{level}' "
                    f"in fused_features. "
                    f"Available keys: {list(fused_features.keys())}"
                )


        self.features = OrderedDict({

            "0": fused_features["p2"],

            "1": fused_features["p3"],

            "2": fused_features["p4"],

            "3": fused_features["p5"]

        })


    def forward(self, images):

        if self.features is None:

            raise RuntimeError(
                "FPN features are not set. "
                "Call set_features() before Faster R-CNN forward."
            )

        return self.features


###############################################################
# SICDA Detector
###############################################################

class SICDADetector(nn.Module):
    """
    SICDA Detection Module.

    This module receives already-fused multi-scale
    RGB-Thermal features from SICDA and performs
    object detection using Faster R-CNN.

    Pipeline:

        SICDA fused features
                |
                v
        p2 / p3 / p4 / p5
                |
                v
          Faster R-CNN
                |
                v
       boxes / labels / scores
    """

    def __init__(self):

        super().__init__()


        ###########################################################
        # Backbone wrapper
        ###########################################################

        self.fpn_wrapper = BackboneWithFPN()


        ###########################################################
        # FLIR-Optimized Anchor Generator
        ###########################################################

        anchor_generator = AnchorGenerator(

            sizes=Config.ANCHOR_SIZES,

            aspect_ratios=Config.ANCHOR_ASPECT_RATIOS

        )


        ###########################################################
        # ROI Pooler
        ###########################################################

        roi_pooler = MultiScaleRoIAlign(

            featmap_names=[
                "0",
                "1",
                "2",
                "3"
            ],

            output_size=7,

            sampling_ratio=2

        )


        ###########################################################
        # Faster R-CNN
        ###########################################################

        self.detector = FasterRCNN(

            backbone=self.fpn_wrapper,

            num_classes=Config.NUM_CLASSES,

            rpn_anchor_generator=anchor_generator,

            box_roi_pool=roi_pooler,

            min_size=Config.MIN_SIZE,

            max_size=Config.MAX_SIZE

        )


    ###############################################################
    # Forward
    ###############################################################

    def forward(
        self,
        fused_features,
        images,
        targets=None
    ):
        """
        Training:
            returns Faster R-CNN loss dictionary.

        Inference:
            returns detections.

        Parameters
        ----------
        fused_features : dict
            SICDA fused FPN features.

        images : list[Tensor]
            Input images.

        targets : list[dict], optional
            Detection targets.
        """


        ###########################################################
        # Set SICDA FPN features
        ###########################################################

        self.fpn_wrapper.set_features(
            fused_features
        )


        ###########################################################
        # Training
        ###########################################################

        if self.training:

            if targets is None:

                raise ValueError(
                    "Targets are required during training."
                )

            return self.detector(
                images,
                targets
            )


        ###########################################################
        # Inference
        ###########################################################

        return self.detector(
            images
        )


    ###############################################################
    # Detection Loss
    ###############################################################

    @staticmethod
    def total_detection_loss(loss_dict):

        total_loss = 0.0

        for loss in loss_dict.values():

            total_loss += loss

        return total_loss


    ###############################################################
    # Prediction
    ###############################################################

    def predict(
        self,
        fused_features,
        images
    ):
        """
        Inference helper.

        Returns:

            list of detection dictionaries
        """

        self.eval()

        with torch.no_grad():

            detections = self.forward(

                fused_features=fused_features,

                images=images,

                targets=None

            )

        return detections


###############################################################
# Factory
###############################################################

def build_detector():

    return SICDADetector()


###############################################################
# Detector Information
###############################################################

def detector_info():

    print()

    print("=" * 70)

    print("SICDA Detector")

    print("=" * 70)

    print("Detector        : Faster R-CNN")

    print("Backbone Input  : SICDA Fused Features")

    print("Feature Levels  : p2, p3, p4, p5")

    print("Classes         :", Config.NUM_CLASSES)

    print("Class Names     :", Config.CLASS_NAMES)

    print("Feature Dim     :", Config.FEATURE_DIM)

    print("Image Min Size  :", Config.MIN_SIZE)

    print("Image Max Size  :", Config.MAX_SIZE)

    print()

    print("Anchor Sizes    :")

    for i, size in enumerate(Config.ANCHOR_SIZES):

        print(
            f"  Level {i}: {size}"
        )

    print()

    print("Anchor Ratios   :")

    for i, ratio in enumerate(
        Config.ANCHOR_ASPECT_RATIOS
    ):

        print(
            f"  Level {i}: {ratio}"
        )

    print()

    print("NMS Threshold   :", Config.NMS_THRESHOLD)

    print("Score Threshold :", Config.SCORE_THRESHOLD)

    print("=" * 70)


###############################################################
# Move Images to Device
###############################################################

def move_images_to_device(
    images,
    device
):

    return [
        img.to(device)
        for img in images
    ]


###############################################################
# Move Targets to Device
###############################################################

def move_targets_to_device(
    targets,
    device
):

    output = []


    for target in targets:

        converted_target = {

            "boxes": target["boxes"].to(device),

            "labels": target["labels"].to(device)

        }


        # Preserve optional fields if available.
        # This makes the detector compatible with
        # additional torchvision target fields.

        if "image_id" in target:

            converted_target["image_id"] = (
                target["image_id"].to(device)
                if torch.is_tensor(target["image_id"])
                else target["image_id"]
            )


        if "area" in target:

            converted_target["area"] = (
                target["area"].to(device)
                if torch.is_tensor(target["area"])
                else target["area"]
            )


        if "iscrowd" in target:

            converted_target["iscrowd"] = (
                target["iscrowd"].to(device)
                if torch.is_tensor(target["iscrowd"])
                else target["iscrowd"]
            )


        output.append(
            converted_target
        )


    return output


###############################################################
# Detector Sanity Check
###############################################################

def test_detector_shapes():

    """
    Lightweight detector construction test.

    This test verifies:

        1. Config can be loaded.
        2. Anchor configuration is valid.
        3. Faster R-CNN can be constructed.
        4. Number of feature levels matches anchors.
    """

    print()

    print("=" * 70)

    print("SICDA Detector Configuration Test")

    print("=" * 70)


    print(
        "Feature levels :",
        len(Config.ANCHOR_SIZES)
    )

    print(
        "Anchor levels  :",
        len(Config.ANCHOR_ASPECT_RATIOS)
    )


    assert len(
        Config.ANCHOR_SIZES
    ) == 4, (
        "SICDA expects 4 FPN levels: "
        "p2, p3, p4, p5."
    )


    assert len(
        Config.ANCHOR_ASPECT_RATIOS
    ) == 4, (
        "There must be 4 aspect-ratio groups "
        "for p2-p5."
    )


    detector = build_detector()


    print(
        "Detector created successfully."
    )

    print(
        "Number of classes:",
        Config.NUM_CLASSES
    )

    print(
        "Feature dimension:",
        Config.FEATURE_DIM
    )

    print(
        "Anchor sizes:",
        Config.ANCHOR_SIZES
    )

    print(
        "Anchor ratios:",
        Config.ANCHOR_ASPECT_RATIOS
    )


    print("=" * 70)

    print(
        "Detector configuration test PASSED"
    )

    print("=" * 70)


###############################################################
# Test
###############################################################

if __name__ == "__main__":

    from torch.utils.data import DataLoader


    ###############################################################
    # SICDA Modules
    ###############################################################

    from backbone import DualBackbone

    from cross_modal_attention import (
        MultiScaleCrossAttention
    )

    from illumination_module import (
        IlluminationAwareFusion
    )


    ###############################################################
    # Collate Functions
    ###############################################################

    from collate import (
        source_collate_fn,
        target_collate_fn
    )


    ###############################################################
    # Transforms
    ###############################################################

    from transforms import (
        build_source_validation_transform,
        build_target_train_transform
    )


    ###############################################################
    # Source Dataset
    ###############################################################

    try:

        from coco import CocoDataset

    except ImportError:

        from coco_dataset import CocoDataset


    ###############################################################
    # Target Dataset
    ###############################################################

    from Flir import FLIRTargetDataset


    ###############################################################
    # Device
    ###############################################################

    device = torch.device(
        Config.DEVICE
    )


    print("=" * 70)

    print("Testing SICDA Detector")

    print("=" * 70)

    print(
        "Device:",
        device
    )


    ###############################################################
    # Detector Configuration Test
    ###############################################################

    test_detector_shapes()


    ###############################################################
    # 1) COCO Source
    ###############################################################
    #
    # COCO is used here only for checking the
    # supervised detector loss.
    #
    # Final target-domain evaluation must be
    # performed on FLIR.
    #
    ###############################################################

    dataset = CocoDataset(

        image_dir=Config.SOURCE_IMAGE_DIR,

        annotation_file=Config.SOURCE_ANNOTATION,

        training=True,

        transforms=build_source_validation_transform()

    )


    dataloader = DataLoader(

        dataset,

        batch_size=1,

        shuffle=False,

        collate_fn=source_collate_fn

    )


    ###############################################################
    # Models
    ###############################################################

    backbone = DualBackbone(
        pretrained=Config.PRETRAINED
    ).to(device)


    attention = MultiScaleCrossAttention(

        channels=Config.FEATURE_DIM,

        num_heads=Config.ATTENTION_HEADS

    ).to(device)


    illumination = IlluminationAwareFusion(

        channels=Config.FEATURE_DIM

    ).to(device)


    detector = build_detector().to(device)


    ###############################################################
    # One COCO Batch
    ###############################################################

    images, targets = next(
        iter(dataloader)
    )


    rgb_images = torch.stack(
        images
    ).to(device)


    ###############################################################
    # COCO has no thermal modality.
    #
    # Synthetic thermal representation is used only
    # for this detector sanity test.
    #
    ###############################################################

    thermal_images = torch.mean(

        rgb_images,

        dim=1,

        keepdim=True

    )


    image_list = [

        img.to(device)

        for img in images

    ]


    targets = move_targets_to_device(

        targets,

        device

    )


    ###############################################################
    # Feature Pipeline
    ###############################################################

    backbone.train()

    attention.train()

    illumination.train()

    detector.train()


    outputs = backbone(

        rgb_images,

        thermal_images

    )


    rgb_features = outputs["rgb"]

    thermal_features = outputs["thermal"]


    attention_features, _ = attention(

        rgb_features,

        thermal_features

    )


    fused_features, alpha = illumination(

        attention_features,

        thermal_features

    )


    ###############################################################
    # Training Loss Test
    ###############################################################

    loss_dict = detector(

        fused_features=fused_features,

        images=image_list,

        targets=targets

    )


    print()

    print("=" * 70)

    print("Training Losses (COCO Source)")

    print("=" * 70)


    total_loss = 0.0


    for name, value in loss_dict.items():

        print(
            f"{name:<25}: "
            f"{value.item():.6f}"
        )

        total_loss += value.item()


    print("-" * 70)

    print(
        f"Total Loss : "
        f"{total_loss:.6f}"
    )


    ###############################################################
    # Evaluation Test
    ###############################################################

    backbone.eval()

    attention.eval()

    illumination.eval()

    detector.eval()


    with torch.no_grad():

        outputs = backbone(

            rgb_images,

            thermal_images

        )


        rgb_features = outputs["rgb"]

        thermal_features = outputs["thermal"]


        attention_features, _ = attention(

            rgb_features,

            thermal_features

        )


        fused_features, alpha = illumination(

            attention_features,

            thermal_features

        )


        detections = detector(

            fused_features=fused_features,

            images=image_list

        )


    ###############################################################
    # Detection Summary
    ###############################################################

    print()

    print("=" * 70)

    print("Detection Summary (COCO Sanity Check)")

    print("=" * 70)


    for i, det in enumerate(detections):

        print(
            f"\nImage {i}"
        )


        print(
            "Detected Boxes :",
            len(det["boxes"])
        )


        if len(det["boxes"]) == 0:

            continue


        for j in range(
            min(5, len(det["boxes"]))
        ):

            label = int(
                det["labels"][j]
            )

            score = float(
                det["scores"][j]
            )

            box = det["boxes"][j]


            if (
                label >= 0
                and
                label < len(Config.CLASS_NAMES)
            ):

                class_name = (
                    Config.CLASS_NAMES[label]
                )

            else:

                class_name = (
                    f"class_{label}"
                )


            print(
                f"{j + 1}. "
                f"{class_name} "
                f"{score:.3f}"
            )


            print(
                "Box:",
                box.cpu().numpy()
            )


    ###############################################################
    # Optional FLIR Target Forward Check
    ###############################################################
    #
    # This checks that real FLIR RGB + thermal
    # data can pass through the SICDA feature pipeline.
    #
    # It does NOT calculate detection loss because
    # this test assumes the target loader is unlabeled.
    #
    ###############################################################

    print()

    print("=" * 70)

    print("Optional FLIR Target Feature Check")

    print("=" * 70)


    try:

        flir_dataset = FLIRTargetDataset(

            rgb_dir=Config.TARGET_RGB_DIR,

            thermal_dir=Config.TARGET_IR_DIR,

            transforms=build_target_train_transform(),

            training=True

        )


        flir_loader = DataLoader(

            flir_dataset,

            batch_size=1,

            shuffle=False,

            collate_fn=target_collate_fn

        )


        flir_batch = next(
            iter(flir_loader)
        )


        flir_rgb = torch.stack(
            flir_batch["rgb"]
        ).to(device)


        flir_thermal = torch.stack(
            flir_batch["thermal"]
        ).to(device)


        with torch.no_grad():

            flir_out = backbone(

                flir_rgb,

                flir_thermal

            )


            flir_rgb_f = flir_out["rgb"]

            flir_th_f = flir_out["thermal"]


            flir_attn, _ = attention(

                flir_rgb_f,

                flir_th_f

            )


            flir_fused, flir_alpha = illumination(

                flir_attn,

                flir_th_f

            )


        print(
            "FLIR RGB batch     :",
            flir_rgb.shape
        )


        print(
            "FLIR Thermal batch :",
            flir_thermal.shape
        )


        print(
            "FLIR fused levels  :",
            list(flir_fused.keys())
        )


        for k, v in flir_fused.items():

            print(
                f"  {k}: "
                f"{tuple(v.shape)}"
            )


        print(
            "FLIR feature check OK"
        )


    except Exception as e:

        print(
            "FLIR optional check skipped / failed:"
        )

        print(
            " ",
            e
        )


    ###############################################################
    # Complete
    ###############################################################

    print()

    print("=" * 70)

    print(
        "Detector Test Completed Successfully"
    )

    print("=" * 70)