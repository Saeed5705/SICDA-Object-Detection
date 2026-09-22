

import random

import torch

import torchvision.transforms.functional as F

from torchvision.transforms import ColorJitter

from config import Config



# ===============================================================
# Compose
# ===============================================================

class Compose:


    def __init__(self, transforms):

        self.transforms = transforms



    def __call__(self, image, target=None):


        for t in self.transforms:

            image, target = t(
                image,
                target
            )


        return image, target




# ===============================================================
# Resize Source Image + Boxes
# ===============================================================


class Resize:


    def __init__(self,
                 size=Config.IMAGE_SIZE):

        self.size=size



    def __call__(
            self,
            image,
            target=None
    ):


        old_w, old_h = image.size



        image = F.resize(
            image,
            (self.size,self.size)
        )



        if target is not None:


            boxes = target["boxes"]



            if len(boxes)>0:


                boxes[:,0] *= self.size/old_w
                boxes[:,2] *= self.size/old_w

                boxes[:,1] *= self.size/old_h
                boxes[:,3] *= self.size/old_h


                target["boxes"]=boxes



                # update area

                target["area"] = (

                    boxes[:,2]-boxes[:,0]

                ) * (

                    boxes[:,3]-boxes[:,1]

                )



        return image,target




# ===============================================================
# Horizontal Flip
# ===============================================================


class RandomHorizontalFlip:


    def __init__(self,p=0.5):

        self.p=p



    def __call__(
            self,
            image,
            target=None
    ):


        if random.random()<self.p:


            width=image.width


            image=F.hflip(image)



            if target is not None:


                boxes=target["boxes"]


                if len(boxes)>0:


                    xmin=boxes[:,0].clone()

                    xmax=boxes[:,2].clone()


                    boxes[:,0]=width-xmax

                    boxes[:,2]=width-xmin



                    target["boxes"]=boxes



        return image,target




# ===============================================================
# RGB Color Augmentation
# ===============================================================


class RGBColorJitter:


    def __init__(self):


        self.jitter=ColorJitter(

            brightness=0.2,

            contrast=0.2,

            saturation=0.2,

            hue=0.05

        )



    def __call__(self,image,target=None):


        image=self.jitter(image)


        return image,target




# ===============================================================
# Tensor
# ===============================================================


class ToTensor:


    def __call__(self,image,target=None):


        image=F.to_tensor(image)


        return image,target




# ===============================================================
# Normalize
# ===============================================================


class Normalize:


    def __call__(self,image,target=None):


        image=F.normalize(

            image,

            mean=Config.PIXEL_MEAN,

            std=Config.PIXEL_STD

        )


        return image,target




# ===============================================================
# Source FLIR-ADAS Training
# ===============================================================


class SourceTrainTransform:


    def __init__(self):


        self.transforms=Compose([


            Resize(),

            RandomHorizontalFlip(),

            RGBColorJitter(),

            ToTensor(),

            Normalize()


        ])




    def __call__(
            self,
            image,
            target
    ):


        return self.transforms(
            image,
            target
        )




# ===============================================================
# Source Validation
# ===============================================================


class SourceValidationTransform:


    def __init__(self):


        self.transforms=Compose([

            Resize(),

            ToTensor(),

            Normalize()

        ])




    def __call__(
            self,
            image,
            target
    ):


        return self.transforms(
            image,
            target
        )




# ===============================================================
# KAIST RGB + Thermal Training
# ===============================================================


class TargetTrainTransform:


    def __init__(self):


        self.rgb_jitter=ColorJitter(

            brightness=0.2,

            contrast=0.2,

            saturation=0.2,

            hue=0.05

        )




    def __call__(
            self,
            rgb,
            thermal
    ):


        # Resize both modalities together

        rgb=F.resize(

            rgb,

            (Config.IMAGE_SIZE,
             Config.IMAGE_SIZE)

        )


        thermal=F.resize(

            thermal,

            (Config.IMAGE_SIZE,
             Config.IMAGE_SIZE)

        )



        # Same flip for RGB and IR


        if random.random()<0.5:


            rgb=F.hflip(rgb)

            thermal=F.hflip(thermal)




        # RGB augmentation only


        rgb=self.rgb_jitter(rgb)




        # Tensor


        rgb=F.to_tensor(rgb)

        thermal=F.to_tensor(thermal)




        # Normalize RGB


        rgb=F.normalize(

            rgb,

            Config.PIXEL_MEAN,

            Config.PIXEL_STD

        )



        # Normalize thermal


        thermal=F.normalize(

            thermal,

            mean=[0.5],

            std=[0.5]

        )



        return rgb,thermal




# ===============================================================
# KAIST Validation
# ===============================================================


class TargetValidationTransform:


    def __call__(
            self,
            rgb,
            thermal
    ):


        rgb=F.resize(

            rgb,

            (Config.IMAGE_SIZE,
             Config.IMAGE_SIZE)

        )


        thermal=F.resize(

            thermal,

            (Config.IMAGE_SIZE,
             Config.IMAGE_SIZE)

        )



        rgb=F.to_tensor(rgb)

        thermal=F.to_tensor(thermal)



        rgb=F.normalize(

            rgb,

            Config.PIXEL_MEAN,

            Config.PIXEL_STD

        )



        thermal=F.normalize(

            thermal,

            mean=[0.5],

            std=[0.5]

        )



        return rgb,thermal




# ===============================================================
# Factory Functions
# ===============================================================


def build_source_train_transform():

    return SourceTrainTransform()



def build_source_validation_transform():

    return SourceValidationTransform()



def build_target_train_transform():

    return TargetTrainTransform()



def build_target_validation_transform():

    return TargetValidationTransform()