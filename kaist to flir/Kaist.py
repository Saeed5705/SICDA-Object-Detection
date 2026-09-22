"""
==============================================================
File : kaist.py

SICDA KAIST RGB-Thermal Dataset Loader

Updated:
- COCO JSON annotation support
- TXT annotation fallback
- RGB + LWIR pair loading
- Faster R-CNN compatible targets
- Person / Car / Bicycle classes

Author:
Muhammad Saeed
==============================================================
"""


import os
import json

import torch

from torch.utils.data import Dataset

from PIL import Image

import torchvision.transforms as T



# ============================================================
# KAIST Dataset
# ============================================================


class KAISTDataset(Dataset):


    def __init__(

            self,

            root,

            split="train",

            transform=None,

            is_source=True

    ):


        self.root = root

        self.split = split

        self.transform = transform

        self.is_source = is_source


        self.samples = []


        self.annotations = {}



        # Annotation paths

        self.annotation_json = os.path.join(

            root,

            "kaist_annotations.JSON"

        )


        self.annotation_txt = os.path.join(

            root,

            "kaist_annotations.txt"

        )


        self.tensor = T.ToTensor()



        # Load annotations

        self.load_annotations()



        # Load image pairs

        self.load_samples()



    # ========================================================
    # Load COCO JSON / TXT Annotation
    # ========================================================


    def load_annotations(self):


        print("="*70)

        print("Loading KAIST annotations")

        print("="*70)



        # ----------------------------------------------------
        # Prefer COCO JSON
        # ----------------------------------------------------

        if os.path.exists(self.annotation_json):


            print(

                "Using JSON:",

                self.annotation_json

            )


            with open(

                self.annotation_json,

                "r",

                encoding="utf-8"

            ) as f:


                data = json.load(f)



            # COCO format

            if (

                isinstance(data, dict)

                and

                "images" in data

                and

                "annotations" in data

            ):



                images = data["images"]

                annotations = data["annotations"]



                image_map = {}



                for img in images:


                    image_map[img["id"]] = os.path.normpath(

                        img["file_name"]

                    )



                for ann in annotations:



                    image_id = ann["image_id"]



                    if image_id not in image_map:

                        continue



                    key = image_map[image_id]



                    record = {


                        "bbox": ann["bbox"],


                        "category_id":

                            ann.get(

                                "category_id",

                                1

                            )


                    }



                    if key not in self.annotations:


                        self.annotations[key] = []



                    self.annotations[key].append(record)



                print(

                    "Images with annotations:",

                    len(self.annotations)

                )



                print("="*70)

                return



        # ----------------------------------------------------
        # TXT fallback
        # ----------------------------------------------------


        if os.path.exists(self.annotation_txt):


            print(

                "Using TXT:",

                self.annotation_txt

            )


            self.load_txt_annotations()



        else:


            print(

                "No annotation file found"

            )



        print("="*70)




    # ========================================================
    # TXT Loader
    # ========================================================


    def load_txt_annotations(self):


        current_image = None


        with open(

            self.annotation_txt,

            "r",

            encoding="utf-8"

        ) as f:


            lines = f.readlines()



        i = 0



        while i < len(lines):


            line = lines[i].strip()



            if line.startswith("Image:"):


                current_image = lines[i+1].strip()


            elif line.startswith("BBox:"):


                bbox_line = lines[i+1].strip()


                bbox = [

                    float(x)

                    for x in bbox_line.split()

                ]



                key = os.path.normpath(

                    os.path.relpath(

                        current_image,

                        self.root

                    )

                )



                if key not in self.annotations:

                    self.annotations[key] = []



                self.annotations[key].append(

                    {

                        "bbox":bbox,

                        "category_id":1

                    }

                )



            i += 1



        print(

            "TXT images with annotations:",

            len(self.annotations)

        )



    # ========================================================
    # Scan KAIST RGB Thermal Pairs
    # ========================================================


    def load_samples(self):


        print("="*70)

        print("Scanning KAIST Dataset")

        print("="*70)



        count = 0



        for set_name in sorted(os.listdir(self.root)):


            if not set_name.startswith("set"):

                continue



            set_path = os.path.join(

                self.root,

                set_name

            )



            if not os.path.isdir(set_path):

                continue



            for video in sorted(os.listdir(set_path)):



                video_path = os.path.join(

                    set_path,

                    video

                )



                visible_dir = os.path.join(

                    video_path,

                    "visible"

                )


                lwir_dir = os.path.join(

                    video_path,

                    "lwir"

                )



                if not os.path.exists(visible_dir):

                    continue



                if not os.path.exists(lwir_dir):

                    continue



                for img in sorted(os.listdir(visible_dir)):


                    if not img.lower().endswith(

                        (".jpg",".jpeg",".png")

                    ):

                        continue



                    rgb_path = os.path.join(

                        visible_dir,

                        img

                    )



                    thermal_path = os.path.join(

                        lwir_dir,

                        img

                    )



                    if not os.path.exists(thermal_path):

                        continue



                    key = os.path.normpath(

                        os.path.relpath(

                            rgb_path,

                            self.root

                        )

                    )



                    self.samples.append(

                        {

                            "rgb":rgb_path,

                            "thermal":thermal_path,

                            "annotation":

                                self.annotations.get(

                                    key,

                                    []

                                )

                        }

                    )



                    count += 1



        print(

            "Total RGB-Thermal pairs:",

            count

        )


        print("="*70)
        # ========================================================
    # Read Annotation
    # ========================================================


    def read_annotation(

            self,

            annotation_list

    ):


        boxes = []

        labels = []



        if annotation_list is None:

            annotation_list = []



        for ann in annotation_list:



            bbox = ann.get(

                "bbox",

                None

            )



            if bbox is None:

                continue



            try:


                x = float(bbox[0])

                y = float(bbox[1])

                w = float(bbox[2])

                h = float(bbox[3])


            except:


                continue



            # COCO format:
            # x,y,w,h
            #
            # Faster RCNN:
            # x1,y1,x2,y2


            boxes.append(

                [

                    x,

                    y,

                    x+w,

                    y+h

                ]

            )



            category = ann.get(

                "category_id",

                1

            )



            # ------------------------------------------------
            # Category Mapping
            # ------------------------------------------------


            if isinstance(category,str):


                name = category.lower()



                if "person" in name:

                    labels.append(1)


                elif "car" in name:

                    labels.append(2)


                elif "bicycle" in name:

                    labels.append(3)


                else:

                    labels.append(1)



            else:


                labels.append(

                    int(category)

                )



        # Empty target handling


        if len(boxes)==0:


            boxes = torch.zeros(

                (0,4),

                dtype=torch.float32

            )


            labels = torch.zeros(

                (0,),

                dtype=torch.long

            )


        else:


            boxes = torch.tensor(

                boxes,

                dtype=torch.float32

            )


            labels = torch.tensor(

                labels,

                dtype=torch.long

            )



        return boxes, labels




    # ========================================================
    # Dataset Length
    # ========================================================


    def __len__(self):


        return len(self.samples)




    # ========================================================
    # Get Item
    # ========================================================


    def __getitem__(

            self,

            idx

    ):



        item = self.samples[idx]



        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------


        rgb = Image.open(

            item["rgb"]

        ).convert(

            "RGB"

        )



        # ----------------------------------------------------
        # Thermal
        # ----------------------------------------------------


        thermal = Image.open(

            item["thermal"]

        ).convert(

            "L"

        )



        # ----------------------------------------------------
        # Annotation
        # ----------------------------------------------------


        boxes, labels = self.read_annotation(

            item["annotation"]

        )



        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------


        if self.transform:


            rgb = self.transform(rgb)

            thermal = self.transform(thermal)


        else:


            rgb = self.tensor(rgb)

            thermal = self.tensor(thermal)



        # ----------------------------------------------------
        # Faster RCNN Target
        # ----------------------------------------------------


        target = {


            "boxes": boxes,


            "labels": labels,


            "image_id":

                torch.tensor(

                    idx,

                    dtype=torch.long

                )

        }



        return {


            "rgb": rgb,


            "thermal": thermal,


            "targets": target,


            "rgb_path": item["rgb"],


            "thermal_path": item["thermal"],


            "domain":

                "source"

                if self.is_source

                else

                "target"

        }




# ============================================================
# Test
# ============================================================


if __name__ == "__main__":



    dataset = KAISTDataset(

        root=r"E:\saeedwork\Kaist",

        is_source=False

    )



    print()

    print("="*70)

    print(

        "Total Samples:",

        len(dataset)

    )

    print("="*70)



    found = False



    for i in range(len(dataset)):



        sample = dataset[i]



        if len(sample["targets"]["boxes"]) > 0:



            print()

            print(

                "FOUND ANNOTATED SAMPLE"

            )


            print(

                "Index:",

                i

            )


            print(

                "RGB:",

                sample["rgb_path"]

            )


            print(

                "Thermal:",

                sample["thermal_path"]

            )


            print(

                "Boxes:"

            )

            print(

                sample["targets"]["boxes"]

            )


            print(

                "Labels:"

            )

            print(

                sample["targets"]["labels"]

            )



            found = True

            break



    if not found:


        print(

            "No annotated sample found"

        )