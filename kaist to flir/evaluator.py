"""
==============================================================
File : utils/evaluator.py

SICDA Evaluation Module

Metrics:
    AP@0.50
    AP@0.75
    AP@0.95
    COCO mAP@0.50:0.95
    Precision
    Recall
    F1-score


Source:
    KAIST

Target:
    FLIR


Author:
    Muhammad Saeed
==============================================================
"""


import torch
import numpy as np



class Evaluator:


    def __init__(

            self,

            num_classes=4,

            iou_threshold=0.50

    ):


        self.num_classes = num_classes

        self.iou_threshold = iou_threshold


        self.class_names = {

            1:"Person",

            2:"Car",

            3:"Bicycle"

        }


        self.reset()



    # ========================================================
    # Reset
    # ========================================================


    def reset(self):


        self.predictions=[]

        self.targets=[]


        self.class_AP={}


        self.map50=0.0

        self.map75=0.0

        self.map95=0.0

        self.map5095=0.0


        self.precision=0.0

        self.recall=0.0

        self.f1_score=0.0




    # ========================================================
    # IoU
    # ========================================================


    def compute_iou(

            self,

            box1,

            box2

    ):


        if isinstance(box1,torch.Tensor):

            box1=box1.cpu().numpy()


        if isinstance(box2,torch.Tensor):

            box2=box2.cpu().numpy()



        x1=max(box1[0],box2[0])

        y1=max(box1[1],box2[1])

        x2=min(box1[2],box2[2])

        y2=min(box1[3],box2[3])



        inter_w=max(0,x2-x1)

        inter_h=max(0,y2-y1)


        intersection=inter_w*inter_h



        area1=max(0,box1[2]-box1[0])*max(
            0,
            box1[3]-box1[1]
        )


        area2=max(0,box2[2]-box2[0])*max(
            0,
            box2[3]-box2[1]
        )


        union=area1+area2-intersection


        if union<=0:

            return 0.0


        return intersection/union




    # ========================================================
    # Update
    # ========================================================


    def update(

            self,

            detections,

            batch

    ):


        if not isinstance(batch,dict):

            return


        if "targets" not in batch:

            return



        for det,target in zip(

                detections,

                batch["targets"]

        ):



            prediction={

                "boxes":
                    det["boxes"].detach().cpu()
                    if len(det["boxes"])>0
                    else torch.empty((0,4)),


                "labels":
                    det["labels"].detach().cpu()
                    if len(det["labels"])>0
                    else torch.empty((0,),dtype=torch.long),


                "scores":
                    det["scores"].detach().cpu()
                    if len(det["scores"])>0
                    else torch.empty((0,))

            }



            ground_truth={

                "boxes":
                    target["boxes"].detach().cpu(),


                "labels":
                    target["labels"].detach().cpu()

            }



            self.predictions.append(prediction)

            self.targets.append(ground_truth)




    # ========================================================
    # AP Calculation
    # ========================================================


    def compute_ap(

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


            pred_boxes=pred["boxes"]

            pred_labels=pred["labels"]

            pred_scores=pred["scores"]


            gt_boxes=target["boxes"]

            gt_labels=target["labels"]



            gt_boxes=gt_boxes[
                gt_labels==class_id
            ]


            pred_mask=pred_labels==class_id


            pred_boxes=pred_boxes[pred_mask]

            pred_scores=pred_scores[pred_mask]



            total_gt += len(gt_boxes)



            if len(pred_boxes)==0:

                continue



            matched=torch.zeros(

                len(gt_boxes),

                dtype=torch.bool

            )



            order=torch.argsort(

                pred_scores,

                descending=True

            )


            pred_boxes=pred_boxes[order]

            pred_scores=pred_scores[order]



            for box,score in zip(

                    pred_boxes,

                    pred_scores

            ):



                best_iou=0

                best_gt=-1



                for idx,gt in enumerate(gt_boxes):


                    iou=self.compute_iou(

                        box,

                        gt

                    )


                    if iou>best_iou:

                        best_iou=iou

                        best_gt=idx



                if (

                    best_iou>=threshold

                    and

                    best_gt>=0

                    and

                    not matched[best_gt]

                ):

                    matched[best_gt]=True

                    tp.append(1)


                else:

                    tp.append(0)



                scores.append(float(score))




        if total_gt==0 or len(scores)==0:

            return 0.0



        order=np.argsort(

            -np.array(scores)

        )


        tp=np.array(tp)[order]


        fp=1-tp


        tp=np.cumsum(tp)

        fp=np.cumsum(fp)



        recall=tp/(total_gt+1e-8)

        precision=tp/(tp+fp+1e-8)



        ap=np.trapezoid(
            precision,

            recall

        )


        return float(ap)




    # ========================================================
    # mAP at IoU
    # ========================================================


    def compute_map(

            self,

            threshold

    ):


        aps=[]


        for cid in range(

                1,

                self.num_classes

        ):


            ap=self.compute_ap(

                cid,

                threshold

            )


            aps.append(ap)



            if abs(threshold-0.50)<1e-6:

                self.class_AP[cid]=ap



        return float(

            np.mean(aps)

        )



    # ========================================================
    # COCO mAP
    # ========================================================


    def compute_map5095(self):


        thresholds=np.arange(

            0.50,

            1.00,

            0.05

        )


        values=[]


        for t in thresholds:


            values.append(

                self.compute_map(float(t))

            )



        return float(

            np.mean(values)

        )



    # ========================================================
    # Precision Recall F1
    # ========================================================


    def compute_precision_recall_f1(self):


        TP=0

        FP=0

        FN=0



        for pred,target in zip(

                self.predictions,

                self.targets

        ):


            pred_boxes=pred["boxes"]

            pred_labels=pred["labels"]


            gt_boxes=target["boxes"]

            gt_labels=target["labels"]



            for cid in range(

                    1,

                    self.num_classes

            ):


                preds=pred_boxes[
                    pred_labels==cid
                ]


                gts=gt_boxes[
                    gt_labels==cid
                ]



                matched=torch.zeros(

                    len(gts),

                    dtype=torch.bool

                )



                for p in preds:


                    best=0

                    idx=-1


                    for i,g in enumerate(gts):


                        iou=self.compute_iou(

                            p,

                            g

                        )


                        if iou>best:

                            best=iou

                            idx=i



                    if best>=0.50 and idx>=0 and not matched[idx]:

                        TP+=1

                        matched[idx]=True


                    else:

                        FP+=1



                FN += len(gts)-matched.sum().item()



        self.precision=TP/(TP+FP+1e-8)

        self.recall=TP/(TP+FN+1e-8)


        self.f1_score=(
            2*self.precision*self.recall
            /
            (self.precision+self.recall+1e-8)
        )


        return {

            "precision":self.precision,

            "recall":self.recall,

            "f1_score":self.f1_score

        }



    # ========================================================
    # Final Evaluation
    # ========================================================


    def evaluate(self):


        self.map50=self.compute_map(0.50)

        self.map75=self.compute_map(0.75)

        self.map95=self.compute_map(0.95)


        self.map5095=self.compute_map5095()



        metrics=self.compute_precision_recall_f1()



        return {


            "mAP50":self.map50,

            "mAP75":self.map75,

            "mAP95":self.map95,

            "mAP5095":self.map5095,


            # compatibility

            "mAP@0.50":self.map50,

            "mAP@0.50:0.95":self.map5095,


            "Precision":metrics["precision"],

            "Recall":metrics["recall"],

            "F1-score":metrics["f1_score"],


            "AP_per_class":self.class_AP

        }




    # ========================================================
    # Print Summary
    # ========================================================


    def print_summary(self):


        results=self.evaluate()



        print("\n"+"="*70)

        print("SICDA Evaluation Summary")

        print("="*70)



        print("\nClass AP@0.50")


        for cid,ap in self.class_AP.items():

            print(

                f"{self.class_names[cid]:12s}: {ap:.4f}"

            )



        print("\nMetrics")

        print("-"*70)


        print(

            f"mAP50      : {results['mAP50']:.4f}"

        )

        print(

            f"mAP75      : {results['mAP75']:.4f}"

        )

        print(

            f"mAP95      : {results['mAP95']:.4f}"

        )

        print(

            f"mAP50:95   : {results['mAP5095']:.4f}"

        )


        print(

            f"Precision  : {results['Precision']:.4f}"

        )


        print(

            f"Recall     : {results['Recall']:.4f}"

        )


        print(

            f"F1-score   : {results['F1-score']:.4f}"

        )


        print("="*70)



        return results