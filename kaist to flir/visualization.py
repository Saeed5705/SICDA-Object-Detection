"""
==============================================================
File : utils/visualization.py

SICDA Visualization Module

Features:
    - Detection visualization
    - Ground Truth vs Prediction
    - Confusion Matrix
    - Batch saving


Author:
    Muhammad Saeed

==============================================================
"""


import os

import torch

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from sklearn.metrics import confusion_matrix

import seaborn as sns




# ============================================================
# CLASS MAP
# ============================================================


CLASS_NAMES = {


    0:"Background",

    1:"Person",

    2:"Car",

    3:"Bicycle"

}




CLASS_COLORS = {


    1:"red",

    2:"blue",

    3:"green"

}




# ============================================================
# Convert Tensor Image
# ============================================================


def tensor_to_image(image):


    if isinstance(image,torch.Tensor):


        image=image.detach().cpu()


        if image.shape[0]==3:

            image=image.permute(

                1,

                2,

                0

            )



        image=image.numpy()



    image=np.clip(

        image,

        0,

        255

    )


    return image.astype(np.uint8)





# ============================================================
# Draw Detection
# ============================================================


def draw_detections(

        image,

        prediction,

        ground_truth=None,

        save_path=None,

        score_threshold=0.3

):



    img=tensor_to_image(image)



    plt.figure(

        figsize=(10,10)

    )


    ax=plt.gca()



    ax.imshow(img)



    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------


    boxes=prediction["boxes"].cpu().numpy()

    labels=prediction["labels"].cpu().numpy()

    scores=prediction["scores"].cpu().numpy()



    for box,label,score in zip(

            boxes,

            labels,

            scores

    ):



        if score < score_threshold:

            continue



        x1,y1,x2,y2=box



        rect=patches.Rectangle(

            (

                x1,

                y1

            ),

            x2-x1,

            y2-y1,


            linewidth=2,


            edgecolor=CLASS_COLORS.get(

                label,

                "yellow"

            ),


            facecolor="none"

        )



        ax.add_patch(rect)



        ax.text(

            x1,

            y1-5,


            f"{CLASS_NAMES.get(label)} {score:.2f}",


            fontsize=10,


            color="white",


            bbox=dict(

                facecolor="black",

                alpha=0.5

            )

        )





    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------


    if ground_truth is not None:



        gt_boxes=ground_truth["boxes"].cpu().numpy()

        gt_labels=ground_truth["labels"].cpu().numpy()



        for box,label in zip(

                gt_boxes,

                gt_labels

        ):



            x1,y1,x2,y2=box



            rect=patches.Rectangle(

                (

                    x1,

                    y1

                ),

                x2-x1,

                y2-y1,


                linewidth=2,


                linestyle="--",


                edgecolor="white",


                facecolor="none"

            )


            ax.add_patch(rect)



            ax.text(

                x1,

                y2+10,


                f"GT {CLASS_NAMES.get(label)}",


                color="white"

            )




    plt.axis("off")




    if save_path is not None:


        os.makedirs(

            os.path.dirname(save_path),

            exist_ok=True

        )


        plt.savefig(

            save_path,

            dpi=300,

            bbox_inches="tight"

        )



    plt.close()





# ============================================================
# Batch Visualization
# ============================================================


def visualize_batch(

        images,

        predictions,

        targets=None,

        save_dir="outputs/detections"

):



    os.makedirs(

        save_dir,

        exist_ok=True

    )



    for i in range(len(images)):



        gt=None



        if targets is not None:

            gt=targets[i]



        save_path=os.path.join(

            save_dir,

            f"image_{i}.png"

        )



        draw_detections(

            images[i],

            predictions[i],

            gt,

            save_path

        )






# ============================================================
# Confusion Matrix
# ============================================================


def generate_confusion_matrix(

        predictions,

        targets,

        save_path="outputs/confusion_matrix.png"

):



    y_true=[]

    y_pred=[]



    for pred,target in zip(

            predictions,

            targets

    ):



        pred_labels=pred["labels"].cpu().numpy()

        gt_labels=target["labels"].cpu().numpy()



        length=max(

            len(pred_labels),

            len(gt_labels)

        )



        pred_labels=list(pred_labels)

        gt_labels=list(gt_labels)



        while len(pred_labels)<length:

            pred_labels.append(0)



        while len(gt_labels)<length:

            gt_labels.append(0)



        y_true.extend(gt_labels)

        y_pred.extend(pred_labels)





    cm=confusion_matrix(

        y_true,

        y_pred,

        labels=[0,1,2,3]

    )




    plt.figure(

        figsize=(8,6)

    )



    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        xticklabels=[

            CLASS_NAMES[i]

            for i in range(4)

        ],

        yticklabels=[

            CLASS_NAMES[i]

            for i in range(4)

        ]

    )



    plt.xlabel(

        "Prediction"

    )


    plt.ylabel(

        "Ground Truth"

    )



    plt.title(

        "SICDA Confusion Matrix"

    )



    plt.tight_layout()



    os.makedirs(

        os.path.dirname(save_path),

        exist_ok=True

    )



    plt.savefig(

        save_path,

        dpi=300

    )



    plt.close()




# ============================================================
# Example
# ============================================================


if __name__=="__main__":


    print(

        "SICDA Visualization Module Loaded Successfully"

    )