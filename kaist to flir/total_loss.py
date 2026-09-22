"""
==============================================================
File : losses/total_loss.py

SICDA Total Loss

L_total =

L_det
+ λ_ssl L_ssl
+ λ_adv L_adv
+ λ_cm L_cm
+ λ_mmd L_mmd


Author:
Muhammad Saeed
==============================================================
"""


import torch
import torch.nn as nn


from config import Config





class TotalLoss(nn.Module):


    def __init__(self):

        super().__init__()


        self.lambda_ssl = Config.LAMBDA_SSL

        self.lambda_adv = Config.LAMBDA_ADV

        self.lambda_cm = Config.LAMBDA_CM

        self.lambda_mmd = Config.LAMBDA_MMD





    # =====================================================
    # Convert loss to Tensor
    # =====================================================

    def process_loss(self, loss, device):


        if loss is None:

            return torch.tensor(
                0.0,
                device=device
            )


        # if tuple/list

        if isinstance(loss,(tuple,list)):

            loss = loss[0]



        # if dictionary

        if isinstance(loss,dict):

            values=[]

            for v in loss.values():

                if torch.is_tensor(v):

                    values.append(v)


            if len(values)>0:

                loss=sum(values)

            else:

                loss=torch.tensor(
                    0.0,
                    device=device
                )



        # if number

        if not torch.is_tensor(loss):

            loss=torch.tensor(
                loss,
                dtype=torch.float32,
                device=device
            )



        return loss





    # =====================================================
    # Forward
    # =====================================================


    def forward(

            self,

            det_loss,

            ssl_loss=None,

            adv_loss=None,

            cm_loss=None,

            mmd_loss=None

    ):



        device = det_loss.device



        # -------------------------------
        # Convert all losses
        # -------------------------------


        det_loss=self.process_loss(
            det_loss,
            device
        )


        ssl_loss=self.process_loss(
            ssl_loss,
            device
        )


        adv_loss=self.process_loss(
            adv_loss,
            device
        )


        cm_loss=self.process_loss(
            cm_loss,
            device
        )


        mmd_loss=self.process_loss(
            mmd_loss,
            device
        )




        # -------------------------------
        # Remove NaN
        # -------------------------------


        det_loss=torch.nan_to_num(
            det_loss
        )

        ssl_loss=torch.nan_to_num(
            ssl_loss
        )

        adv_loss=torch.nan_to_num(
            adv_loss
        )

        cm_loss=torch.nan_to_num(
            cm_loss
        )

        mmd_loss=torch.nan_to_num(
            mmd_loss
        )





        # -------------------------------
        # Total SICDA Loss
        # -------------------------------


        total_loss=(

            det_loss

            +

            self.lambda_ssl * ssl_loss

            +

            self.lambda_adv * adv_loss

            +

            self.lambda_cm * cm_loss

            +

            self.lambda_mmd * mmd_loss

        )





        # -------------------------------
        # Logging
        # -------------------------------


        loss_dict={


            "total_loss":

                total_loss.detach(),


            "det_loss":

                det_loss.detach(),


            "ssl_loss":

                ssl_loss.detach(),


            "adv_loss":

                adv_loss.detach(),


            "cm_loss":

                cm_loss.detach(),


            "mmd_loss":

                mmd_loss.detach()



        }




        return total_loss, loss_dict






# ============================================================
# Factory
# ============================================================


def build_total_loss():


    return TotalLoss()





# ============================================================
# Test
# ============================================================


if __name__=="__main__":


    device=torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"

    )



    criterion=build_total_loss().to(device)



    det=torch.tensor(
        1.0,
        device=device,
        requires_grad=True
    )


    ssl=(torch.tensor(
        0.5,
        device=device,
        requires_grad=True
    ),"extra")



    adv=torch.tensor(
        0.2,
        device=device,
        requires_grad=True
    )



    cm=torch.tensor(
        0.1,
        device=device,
        requires_grad=True
    )


    mmd=torch.tensor(
        0.3,
        device=device,
        requires_grad=True
    )



    total,losses=criterion(

        det,
        ssl,
        adv,
        cm,
        mmd

    )



    print("="*60)

    print("SICDA Total Loss Test")

    print("="*60)


    print(
        "Total:",
        total.item()
    )


    for k,v in losses.items():

        print(
            k,
            ":",
            v.item()
        )


    print("="*60)