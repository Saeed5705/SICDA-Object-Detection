"""
==============================================================
File : datasets/transforms.py

SICDA Data Transformation Module

Supports:
    - RGB images
    - Thermal / Infrared images
    - KAIST Dataset
    - FLIR Dataset
    - Faster R-CNN format

Author:
Muhammad Saeed
==============================================================
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import random


# ============================================================
# SICDA Transform Class
# ============================================================

class SICDATransform:

    def __init__(
            self,
            train=True,
            image_size=512
    ):

        self.train = train

        self.image_size = image_size


        # ----------------------------------------------------
        # RGB Normalization (ImageNet)
        # ----------------------------------------------------

        self.rgb_normalize = T.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225]

        )


        # ----------------------------------------------------
        # Thermal Normalization
        # ----------------------------------------------------

        self.thermal_normalize = T.Normalize(

            mean=[0.5],

            std=[0.5]

        )


    # ========================================================
    # Resize Image and Bounding Boxes
    # ========================================================

    def resize(
            self,
            image,
            target
    ):

        _, h, w = image.shape


        scale_x = self.image_size / w

        scale_y = self.image_size / h


        image = TF.resize(

            image,

            [
                self.image_size,
                self.image_size
            ]

        )


        if target is not None:

            boxes = target["boxes"]


            boxes = boxes.clone()


            boxes[:,0] *= scale_x
            boxes[:,2] *= scale_x

            boxes[:,1] *= scale_y
            boxes[:,3] *= scale_y


            target["boxes"] = boxes


        return image, target



    # ========================================================
    # Horizontal Flip
    # ========================================================

    def horizontal_flip(
            self,
            image,
            target
    ):


        if random.random() < 0.5:


            image = torch.flip(

                image,

                dims=[2]

            )


            if target is not None:


                boxes = target["boxes"]


                _, _, width = image.shape


                boxes = boxes.clone()


                x1 = boxes[:,0]

                x2 = boxes[:,2]


                boxes[:,0] = width - x2

                boxes[:,2] = width - x1


                target["boxes"] = boxes


        return image, target



    # ========================================================
    # RGB Processing
    # ========================================================

    def process_rgb(
            self,
            image,
            target=None
    ):


        if not torch.is_tensor(image):

            image = TF.to_tensor(image)


        image, target = self.resize(

            image,

            target

        )


        if self.train:

            image, target = self.horizontal_flip(

                image,

                target

            )


        image = self.rgb_normalize(

            image

        )


        return image, target



    # ========================================================
    # Thermal Processing
    # ========================================================

    def process_thermal(
            self,
            image,
            target=None
    ):


        if not torch.is_tensor(image):

            image = TF.to_tensor(image)


        # Ensure single channel

        if image.shape[0] != 1:


            image = image.mean(

                dim=0,

                keepdim=True

            )



        image, target = self.resize(

            image,

            target

        )


        if self.train:


            image, target = self.horizontal_flip(

                image,

                target

            )


        image = self.thermal_normalize(

            image

        )


        return image, target



    # ========================================================
    # Joint RGB + Thermal Transform
    # ========================================================

    def __call__(
            self,
            rgb,
            thermal,
            target=None
    ):


        rgb, target = self.process_rgb(

            rgb,

            target

        )


        thermal, _ = self.process_thermal(

            thermal,

            target

        )


        return {

            "rgb": rgb,

            "thermal": thermal,

            "target": target

        }



# ============================================================
# Factory Functions
# ============================================================


def get_train_transform():

    return SICDATransform(

        train=True,

        image_size=512

    )



def get_val_transform():

    return SICDATransform(

        train=False,

        image_size=512

    )



# ============================================================
# Test
# ============================================================

if __name__ == "__main__":


    transform = get_train_transform()


    rgb = torch.randn(

        3,

        800,

        800

    )


    thermal = torch.randn(

        1,

        800,

        800

    )


    target = {


        "boxes": torch.tensor(

            [

                [100,100,300,300]

            ],

            dtype=torch.float32

        ),


        "labels": torch.tensor(

            [1],

            dtype=torch.int64

        )

    }



    output = transform(

        rgb,

        thermal,

        target

    )


    print("="*60)

    print("SICDA Transform Test")

    print("="*60)


    print(

        "RGB:",

        output["rgb"].shape

    )


    print(

        "Thermal:",

        output["thermal"].shape

    )


    print(

        "Boxes:",

        output["target"]["boxes"]

    )


    print("="*60)

    print("Transform Test Passed")