"""
==============================================================
File : flir.py

SICDA Target Dataset Loader

Target Domain:
    FLIR ADAS RGB-Thermal Dataset

Classes:
    0 Background
    1 Person
    2 Car
    3 Bicycle

Author:
Muhammad Saeed
==============================================================
"""


import os
import torch

from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms as T

from pycocotools.coco import COCO



class FLIRDataset(Dataset):


    def __init__(
            self,
            root,
            split="train",
            annotation=None,
            transform=None,
            is_source=False
    ):


        self.root=root
        self.split=split
        self.transform=transform
        self.is_source=is_source

        self.tensor=T.ToTensor()



        # ===============================
        # Paths
        # ===============================


        if split=="train":


            self.rgb_root=os.path.join(
                root,
                "images_rgb_train",
                "data"
            )


            self.thermal_root=os.path.join(
                root,
                "images_thermal_train",
                "data"
            )


        else:


            self.rgb_root=os.path.join(
                root,
                "images_rgb_val",
                "data"
            )


            self.thermal_root=os.path.join(
                root,
                "images_thermal_val",
                "data"
            )



        print("="*70)
        print("Loading FLIR Dataset")
        print("="*70)


        print("RGB Folder:",self.rgb_root)
        print("Thermal Folder:",self.thermal_root)



        # ===============================
        # Thermal index
        # ===============================


        self.thermal_map={}


        thermal_files=os.listdir(
            self.thermal_root
        )


        print(
            "Thermal Files:",
            len(thermal_files)
        )


        for f in thermal_files:


            if "frame-" in f:


                frame_id=f.split(
                    "frame-"
                )[1].split("-")[0]


                self.thermal_map[frame_id]=f




        # ===============================
        # COCO
        # ===============================


        self.coco=None


        if annotation and os.path.exists(annotation):


            print(
                "Annotation:",
                annotation
            )


            self.coco=COCO(annotation)


        else:

            print(
                "No COCO annotation found"
            )



        self.samples=[]


        self.load_samples()





    # =====================================
    # Load paired images
    # =====================================


    def load_samples(self):


        rgb_count=0
        thermal_count=0


        files=os.listdir(
            self.rgb_root
        )


        print(
            "RGB Files:",
            len(files)
        )


        for file in sorted(files):


            if not file.lower().endswith(
                (
                ".jpg",
                ".jpeg",
                ".png",
                ".tif",
                ".tiff"
                )
            ):

                continue



            rgb_path=os.path.join(
                self.rgb_root,
                file
            )



            thermal_path=None



            # -------------------------
            # Match thermal using frame
            # -------------------------


            if "frame-" in file:


                frame_id=file.split(
                    "frame-"
                )[1].split("-")[0]



                if frame_id in self.thermal_map:


                    thermal_path=os.path.join(
                        self.thermal_root,
                        self.thermal_map[frame_id]
                    )



            if thermal_path is None:

                continue




            # -------------------------
            # COCO image id
            # -------------------------


            image_id=None


            if self.coco:


                for k,v in self.coco.imgs.items():


                    coco_name=os.path.basename(
                        v["file_name"]
                    )


                    if coco_name==file:


                        image_id=k
                        break





            self.samples.append(
                {

                "rgb":rgb_path,

                "thermal":thermal_path,

                "image_id":image_id

                }
            )


            rgb_count+=1
            thermal_count+=1




        print()

        print("RGB Images:",rgb_count)

        print("Thermal Images:",thermal_count)

        print(
            "RGB-Thermal Pairs:",
            len(self.samples)
        )

        print("="*70)






    # =====================================
    # Annotation
    # =====================================


    def get_annotation(self,image_id):


        boxes=[]
        labels=[]


        if self.coco is None or image_id is None:


            return (
                torch.zeros((0,4)),
                torch.zeros((0,),dtype=torch.long)
            )



        ann_ids=self.coco.getAnnIds(
            imgIds=[image_id]
        )


        anns=self.coco.loadAnns(
            ann_ids
        )



        for ann in anns:


            x,y,w,h=ann["bbox"]


            boxes.append(
                [
                x,
                y,
                x+w,
                y+h
                ]
            )


            cat=ann["category_id"]


            if cat in [1,2,3]:

                labels.append(cat)

            else:

                labels.append(1)



        return (

            torch.tensor(
                boxes,
                dtype=torch.float32
            ),


            torch.tensor(
                labels,
                dtype=torch.long
            )

        )





    def __len__(self):

        return len(self.samples)




    def __getitem__(self,idx):


        item=self.samples[idx]


        rgb=Image.open(
            item["rgb"]
        ).convert("RGB")


        thermal=Image.open(
            item["thermal"]
        ).convert("L")



        boxes,labels=self.get_annotation(
            item["image_id"]
        )



        if self.transform:


            rgb=self.transform(rgb)

            thermal=self.transform(thermal)


        else:


            rgb=self.tensor(rgb)

            thermal=self.tensor(thermal)




        target={

            "boxes":boxes,

            "labels":labels,

            "image_id":torch.tensor(idx)

        }



        return {

            "rgb":rgb,

            "thermal":thermal,

            "targets":target,

            "rgb_path":item["rgb"],

            "thermal_path":item["thermal"],

            "domain":"target"

        }





# ==========================================
# Test
# ==========================================


if __name__=="__main__":


    dataset=FLIRDataset(

        root=r"E:\saeedwork\FLIR_dataset\FLIR_ADAS_v2",

        split="train",

        annotation=r"E:\saeedwork\FLIR_dataset\FLIR_ADAS_v2\images_rgb_train\coco.JSON"

    )


    print()

    print("="*70)

    print("FLIR Dataset Test")

    print("="*70)


    print(
        "Total Samples:",
        len(dataset)
    )


    if len(dataset)>0:


        sample=dataset[0]


        print()

        print(
            "RGB:",
            sample["rgb"].shape
        )


        print(
            "Thermal:",
            sample["thermal"].shape
        )


        print(
            "Boxes:",
            sample["targets"]["boxes"].shape
        )


        print(
            "Labels:",
            sample["targets"]["labels"]
        )


    print("="*70)