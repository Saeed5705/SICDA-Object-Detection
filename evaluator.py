"""
================================================================
SICDA Evaluator

Metrics:
    - mAP@0.50
    - mAP@0.75
    - mAP@0.95
    - mAP@0.50:0.95
    - Precision
    - Recall
    - F1-score

Additional Analysis:
    - Sensitivity Analysis
    - Specificity
    - FPR
    - FNR
    - Computational Complexity
    - Parameters
    - Model Size
    - FPS
    - GPU Memory

Author:
Muhammad Saeed
================================================================
"""


import os
import time
import numpy as np
import torch


class Evaluator:


    def __init__(
            self,
            num_classes=4,
            iou_threshold=0.50,
            model=None,
            device="cuda",
            save_dir="./outputs"
    ):

        self.num_classes = num_classes
        self.iou_threshold = iou_threshold

        self.model = model
        self.device = device
        self.save_dir = save_dir

        os.makedirs(
            self.save_dir,
            exist_ok=True
        )


        self.iou_thresholds = np.arange(
            0.50,
            1.00,
            0.05
        )

        self.reset()



    # ==========================================================
    # Reset
    # ==========================================================

    def reset(self):

        self.predictions = []
        self.targets = []

        self.class_AP = {}
        self.class_AP75 = {}
        self.class_AP95 = {}
        self.class_AP_coco = {}

        self.map50 = 0
        self.map75 = 0
        self.map95 = 0
        self.map = 0

        self.precision = 0
        self.recall = 0
        self.f1_score = 0



    # ==========================================================
    # IoU
    # ==========================================================

    def compute_iou(self, box1, box2):

        if torch.is_tensor(box1):
            box1 = box1.cpu().numpy()

        if torch.is_tensor(box2):
            box2 = box2.cpu().numpy()


        x1=max(box1[0],box2[0])
        y1=max(box1[1],box2[1])

        x2=min(box1[2],box2[2])
        y2=min(box1[3],box2[3])


        inter=max(0,x2-x1)*max(0,y2-y1)


        area1=(box1[2]-box1[0])*(box1[3]-box1[1])
        area2=(box2[2]-box2[0])*(box2[3]-box2[1])


        union=area1+area2-inter


        if union<=0:
            return 0


        return inter/union



    # ==========================================================
    # Update
    # ==========================================================

    def update(self,detections,batch):

        if "targets" not in batch:
            return


        for det,target in zip(
                detections,
                batch["targets"]
        ):


            self.predictions.append({

                "boxes":
                det["boxes"].detach().cpu(),

                "labels":
                det["labels"].detach().cpu(),

                "scores":
                det["scores"].detach().cpu()

            })


            self.targets.append({

                "boxes":
                target["boxes"].detach().cpu(),

                "labels":
                target["labels"].detach().cpu()

            })



    # ==========================================================
    # AP
    # ==========================================================

    def compute_ap_at_iou(
            self,
            class_id,
            threshold
    ):

        scores=[]
        tp=[]
        total_gt=0


        for pred,target in zip(
                self.predictions,
                self.targets
        ):

            pbox=pred["boxes"]
            plabel=pred["labels"]
            pscore=pred["scores"]


            gtbox=target["boxes"]
            gtlabel=target["labels"]



            gtbox=gtbox[
                gtlabel==class_id
            ]

            p_mask=plabel==class_id

            pbox=pbox[p_mask]
            pscore=pscore[p_mask]


            total_gt+=len(gtbox)


            matched=torch.zeros(
                len(gtbox),
                dtype=torch.bool
            )


            order=torch.argsort(
                pscore,
                descending=True
            )


            for idx in order:

                box=pbox[idx]

                best=0
                best_id=-1


                for i,g in enumerate(gtbox):

                    iou=self.compute_iou(
                        box,
                        g
                    )

                    if iou>best:
                        best=iou
                        best_id=i



                if (
                    best>=threshold
                    and best_id>=0
                    and not matched[best_id]
                ):

                    matched[best_id]=True
                    tp.append(1)

                else:
                    tp.append(0)


                scores.append(
                    float(pscore[idx])
                )


        if total_gt==0:
            return 0


        order=np.argsort(
            -np.array(scores)
        )


        tp=np.array(tp)[order]

        fp=1-tp


        tp=np.cumsum(tp)
        fp=np.cumsum(fp)


        recall=tp/(total_gt+1e-8)

        precision=tp/(tp+fp+1e-8)


        return float(
            np.trapz(
                precision,
                recall
            )
        )



    # ==========================================================
    # mAP
    # ==========================================================

    def compute_map(self):

        ap50=[]
        ap75=[]
        ap95=[]
        apall=[]


        for c in range(1,self.num_classes):

            a50=self.compute_ap_at_iou(c,0.50)
            a75=self.compute_ap_at_iou(c,0.75)
            a95=self.compute_ap_at_iou(c,0.95)


            acoco=np.mean([

                self.compute_ap_at_iou(
                    c,t
                )

                for t in self.iou_thresholds

            ])


            self.class_AP[c]=a50
            self.class_AP75[c]=a75
            self.class_AP95[c]=a95
            self.class_AP_coco[c]=acoco


            ap50.append(a50)
            ap75.append(a75)
            ap95.append(a95)
            apall.append(acoco)



        self.map50=np.mean(ap50)
        self.map75=np.mean(ap75)
        self.map95=np.mean(ap95)
        self.map=np.mean(apall)



    # ==========================================================
    # Precision Recall
    # ==========================================================

    def compute_precision_recall(self):

        TP=0
        FP=0
        FN=0


        for pred,target in zip(
                self.predictions,
                self.targets
        ):


            matched=torch.zeros(
                len(target["boxes"]),
                dtype=torch.bool
            )


            for box in pred["boxes"]:

                best=0
                idx=-1


                for i,g in enumerate(target["boxes"]):

                    iou=self.compute_iou(
                        box,g
                    )


                    if iou>best:
                        best=iou
                        idx=i



                if best>=0.5 and idx>=0:

                    TP+=1
                    matched[idx]=True

                else:

                    FP+=1



            FN+=(
                len(target["boxes"])
                -
                matched.sum().item()
            )



        precision=TP/(TP+FP+1e-8)

        recall=TP/(TP+FN+1e-8)

        f1=2*precision*recall/(precision+recall+1e-8)


        return precision,recall,f1,TP,FP,FN



    # ==========================================================
    # Sensitivity Analysis
    # ==========================================================

    def sensitivity_analysis(self):

        precision,recall,f1,TP,FP,FN = self.compute_precision_recall()


        sensitivity=recall


        specificity=TP/(TP+FP+1e-8)


        fpr=FP/(TP+FP+1e-8)

        fnr=FN/(TP+FN+1e-8)



        print("\nSensitivity Analysis")
        print("="*60)

        print(
            f"Sensitivity (Recall/TPR): {sensitivity:.4f}"
        )

        print(
            f"Specificity             : {specificity:.4f}"
        )

        print(
            f"False Positive Rate     : {fpr:.4f}"
        )

        print(
            f"False Negative Rate     : {fnr:.4f}"
        )



    # ==========================================================
    # Complexity
    # ==========================================================

    def complexity_analysis(self):

        if self.model is None:
            return


        params=sum(
            p.numel()
            for p in self.model.parameters()
        )


        trainable=sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )


        size=params*4/(1024**2)


        print("\nComputational Complexity")
        print("="*60)

        print(
            f"Parameters          : {params/1e6:.2f} M"
        )

        print(
            f"Trainable Params    : {trainable/1e6:.2f} M"
        )

        print(
            f"Model Size          : {size:.2f} MB"
        )


        if torch.cuda.is_available():

            print(
                f"GPU Memory          : "
                f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
            )



    # ==========================================================
    # Evaluate
    # ==========================================================

    def evaluate(self):

        self.compute_map()


        precision,recall,f1,_,_,_=self.compute_precision_recall()


        self.precision=precision
        self.recall=recall
        self.f1_score=f1


        self.sensitivity_analysis()

        self.complexity_analysis()



        results={

            "mAP":self.map50,

            "mAP50":self.map50,

            "mAP75":self.map75,

            "mAP95":self.map95,

            "mAP50_95":self.map,


            "Precision":precision,

            "Recall":recall,

            "F1-score":f1,


            "AP_per_class":self.class_AP,

            "AP75_per_class":self.class_AP75,

            "AP95_per_class":self.class_AP95,

            "AP50_95_per_class":self.class_AP_coco

        }


        return results