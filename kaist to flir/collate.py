"""
==============================================================
File : collate.py

SICDA Collate Functions

Domain Setup:

Source:
    KAIST RGB-Thermal
    (with annotations)

Target:
    FLIR RGB-Thermal
    (with annotations)

Training:
    KAIST:
        Detection Loss

    FLIR:
        Domain Adaptation
        MMD Alignment

Evaluation:
    Both annotations supported

Author:
    Muhammad Saeed
==============================================================
"""


import torch
import torch.nn.functional as F



# ============================================================
# Pad Images
# ============================================================


def pad_images(images):

    """
    Convert variable size images
    into batch tensor
    """

    max_h = max(
        img.shape[1]
        for img in images
    )

    max_w = max(
        img.shape[2]
        for img in images
    )


    batch = []


    for img in images:

        c,h,w = img.shape


        pad_h = max_h - h
        pad_w = max_w - w


        img = F.pad(
            img,
            (
                0,
                pad_w,
                0,
                pad_h
            )
        )


        batch.append(img)



    return torch.stack(batch)



# ============================================================
# Target Processing
# ============================================================


def process_targets(targets):

    """
    Validate Faster-RCNN targets
    """


    processed = []


    for target in targets:


        if target is None:

            target = {}



        boxes = target.get(

            "boxes",

            torch.zeros(
                (0,4),
                dtype=torch.float32
            )

        )



        labels = target.get(

            "labels",

            torch.zeros(
                (0,),
                dtype=torch.int64
            )

        )



        processed.append(

            {

                "boxes": boxes.float(),

                "labels": labels.long()

            }

        )



    return processed





# ============================================================
# Common Collate
# ============================================================


def build_collate(batch, domain):


    rgb = [

        item["rgb"]

        for item in batch

    ]


    thermal = [

        item["thermal"]

        for item in batch

    ]


    targets = [

        item.get(
            "targets",
            {}
        )

        for item in batch

    ]



    output = {


        # Tensor batch
        "rgb":

            pad_images(rgb),



        "thermal":

            pad_images(thermal),



        # Faster RCNN format
        "images":

            rgb,



        "targets":

            process_targets(targets),



        "rgb_path":

            [

                item.get(
                    "rgb_path",
                    ""
                )

                for item in batch

            ],



        "thermal_path":

            [

                item.get(
                    "thermal_path",
                    ""
                )

                for item in batch

            ],



        "domain":

            domain

    }



    return output





# ============================================================
# KAIST Source Collate
# ============================================================


def kaist_collate_fn(batch):

    """
    Source Domain

    KAIST
    """

    return build_collate(

        batch,

        domain="source"

    )





# ============================================================
# FLIR Target Collate
# ============================================================


def flir_collate_fn(batch):

    """
    Target Domain

    FLIR
    """

    return build_collate(

        batch,

        domain="target"

    )





# ============================================================
# Joint SICDA Collate
# ============================================================


def sicda_collate_fn(batch):

    """
    Mixed source-target batch
    """


    source = []

    target = []



    for item in batch:


        if item.get("domain") == "source":

            source.append(item)


        else:

            target.append(item)



    output = {}



    if len(source)>0:

        output["source"] = kaist_collate_fn(

            source

        )



    if len(target)>0:

        output["target"] = flir_collate_fn(

            target

        )



    return output





# ============================================================
# Move Batch To Device
# ============================================================


def move_batch_to_device(batch, device):


    if "rgb" in batch:

        batch["rgb"] = batch["rgb"].to(device)



    if "thermal" in batch:

        batch["thermal"] = batch["thermal"].to(device)




    if "targets" in batch:


        for target in batch["targets"]:


            target["boxes"] = target["boxes"].to(device)

            target["labels"] = target["labels"].to(device)



    return batch





# ============================================================
# Test
# ============================================================


if __name__=="__main__":


    sample = {


        "rgb":

            torch.randn(
                3,
                512,
                512
            ),


        "thermal":

            torch.randn(
                1,
                512,
                512
            ),


        "targets":

            {

            "boxes":

                torch.tensor(
                    [
                    [10,20,100,150]
                    ],
                    dtype=torch.float32
                ),


            "labels":

                torch.tensor(
                    [1],
                    dtype=torch.long
                )

            },


        "rgb_path":"test.jpg",

        "thermal_path":"test.png"

    }



    batch = kaist_collate_fn(

        [
            sample,
            sample
        ]

    )


    print("="*60)

    print("Collate Test")

    print("="*60)

    print(
        "RGB:",
        batch["rgb"].shape
    )

    print(
        "Thermal:",
        batch["thermal"].shape
    )

    print(
        "Images:",
        len(batch["images"])
    )

    print(
        "Targets:",
        len(batch["targets"])
    )

    print(
        "Domain:",
        batch["domain"]
    )

    print("="*60)