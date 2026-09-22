"""
==============================================================
File : detector.py

SICDA Detection Head

FPN Features
      |
      |
 RPN + ROI Head
      |
 Detection

==============================================================
"""


import torch
import torch.nn as nn


from torchvision.models.detection.rpn import (
    RegionProposalNetwork,
    RPNHead,
    AnchorGenerator
)


from torchvision.models.detection.roi_heads import (
    RoIHeads
)


from torchvision.ops import (
    MultiScaleRoIAlign
)


from torchvision.models.detection.image_list import (
    ImageList
)


from torchvision.models.detection.faster_rcnn import (
    TwoMLPHead,
    FastRCNNPredictor
)




# ============================================================
# SICDA Detector
# ============================================================


class SICDADetector(nn.Module):


    def __init__(self,num_classes=4):

        super().__init__()


        self.out_channels=256



        # -------------------------
        # Anchor
        # -------------------------

        self.anchor_generator=AnchorGenerator(

            sizes=(

                (32,64,128),
                (64,128,256),
                (128,256,512),
                (256,512,1024)

            ),

            aspect_ratios=(

                (0.5,1,2),
                (0.5,1,2),
                (0.5,1,2),
                (0.5,1,2)

            )

        )



        # -------------------------
        # RPN
        # -------------------------


        rpn_head=RPNHead(

            256,

            self.anchor_generator.num_anchors_per_location()[0]

        )


        self.rpn=RegionProposalNetwork(

            self.anchor_generator,

            rpn_head,

            fg_iou_thresh=0.7,

            bg_iou_thresh=0.3,

            batch_size_per_image=256,

            positive_fraction=0.5,

            pre_nms_top_n={

                "training":2000,

                "testing":1000

            },

            post_nms_top_n={

                "training":1000,

                "testing":300

            },

            nms_thresh=0.7

        )




        # -------------------------
        # ROI Head
        # -------------------------


        roi_pooler=MultiScaleRoIAlign(

            featmap_names=[

                "p2",
                "p3",
                "p4",
                "p5"

            ],

            output_size=7,

            sampling_ratio=2

        )



        box_head=TwoMLPHead(

            256*7*7,

            1024

        )



        box_predictor=FastRCNNPredictor(

            1024,

            num_classes

        )



        self.roi_heads=RoIHeads(

            box_roi_pool=roi_pooler,

            box_head=box_head,

            box_predictor=box_predictor,

            fg_iou_thresh=0.5,

            bg_iou_thresh=0.5,

            batch_size_per_image=512,

            positive_fraction=0.25,

            bbox_reg_weights=None,

            score_thresh=0.05,

            nms_thresh=0.5,

            detections_per_img=100

        )




    # ========================================================
    # Forward
    # ========================================================


    def forward(

            self,

            features,

            images,

            targets=None

    ):


        """

        features:

        {
        p2,
        p3,
        p4,
        p5
        }


        images:

        list of tensors


        """



        image_sizes=[

            img.shape[-2:]

            for img in images

        ]



        tensors=torch.stack(images)



        image_list=ImageList(

            tensors,

            image_sizes

        )




        proposals, rpn_losses=self.rpn(

            image_list,

            features,

            targets

        )




        detections, roi_losses=self.roi_heads(

            features,

            proposals,

            image_list.image_sizes,

            targets

        )



        losses={}

        losses.update(rpn_losses)

        losses.update(roi_losses)



        if self.training:

            return losses


        return detections





# ============================================================
# Factory
# ============================================================


def build_detector(num_classes=4):

    return SICDADetector(

        num_classes

    )






# ============================================================
# Test
# ============================================================


if __name__=="__main__":


    device=torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"

    )


    detector=build_detector().to(device)



    features={

        "p2":torch.randn(2,256,128,128,device=device),

        "p3":torch.randn(2,256,64,64,device=device),

        "p4":torch.randn(2,256,32,32,device=device),

        "p5":torch.randn(2,256,16,16,device=device)

    }



    images=[

        torch.randn(3,512,512,device=device),

        torch.randn(3,512,512,device=device)

    ]



    detector.eval()


    with torch.no_grad():

        output=detector(

            features,

            images

        )


    print("="*60)

    print("SICDA Detector Test")

    print("="*60)


    print(

        len(output)

    )


    print(output[0].keys())


    print("Detector Passed")
