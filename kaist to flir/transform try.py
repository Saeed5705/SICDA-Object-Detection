import random

import torch
import torchvision.transforms.functional as TF

from torchvision.transforms import InterpolationMode



# ============================================================
# KAIST / FLIR RGB-T Transform
# ============================================================


class RGBTTTransform:


    def __init__(
        self,
        image_size=(512,512),
        train=True
    ):

        self.image_size = image_size
        self.train = train



    def __call__(
        self,
        rgb,
        thermal,
        target=None
    ):


        # --------------------------------
        # Resize
        # --------------------------------

        rgb = TF.resize(
            rgb,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR
        )


        thermal = TF.resize(
            thermal,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR
        )



        # --------------------------------
        # Horizontal Flip
        # Apply same flip on both modalities
        # --------------------------------

        if self.train:

            if random.random() < 0.5:


                rgb = TF.hflip(
                    rgb
                )


                thermal = TF.hflip(
                    thermal
                )


                # update bounding boxes

                if target is not None and "boxes" in target:


                    width = self.image_size[1]


                    boxes = target["boxes"]


                    if boxes.numel() > 0:


                        boxes[:, [0,2]] = (
                            width -
                            boxes[:, [2,0]]
                        )


                        target["boxes"] = boxes



        # --------------------------------
        # Convert to tensor
        # --------------------------------

        rgb = TF.to_tensor(
            rgb
        )


        thermal = TF.to_tensor(
            thermal
        )



        # --------------------------------
        # RGB normalization
        # --------------------------------

        rgb = TF.normalize(

            rgb,

            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]

        )



        # --------------------------------
        # Thermal normalization
        # --------------------------------

        thermal = TF.normalize(

            thermal,

            mean=[0.5],

            std=[0.5]

        )



        if target is not None:

            return (
                rgb,
                thermal,
                target
            )


        else:

            return (
                rgb,
                thermal
            )





# ============================================================
# Train Transform
# ============================================================


def get_train_transform():

    return RGBTTTransform(

        image_size=(512,512),

        train=True

    )





# ============================================================
# Validation Transform
# ============================================================


def get_val_transform():

    return RGBTTTransform(

        image_size=(512,512),

        train=False

    )