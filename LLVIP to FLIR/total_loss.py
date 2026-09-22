"""
==============================================================
File : total_loss.py

Total Loss for SICDA Framework

L_total =
    L_det
    + λ1 L_ssl
    + λ2 L_adv
    + λ3 L_cm
    + λ4 L_mmd

Author:
Muhammad Saeed
==============================================================
"""

import torch
import torch.nn as nn


class TotalLoss(nn.Module):

    def __init__(

            self,

            lambda_ssl=1.0,

            lambda_adv=0.5,

            lambda_cm=0.2,

            lambda_mmd=0.1

    ):

        super().__init__()

        self.lambda_ssl = lambda_ssl
        self.lambda_adv = lambda_adv
        self.lambda_cm = lambda_cm
        self.lambda_mmd = lambda_mmd


    def forward(

            self,

            det_loss,

            ssl_loss,

            adv_loss,

            cm_loss,

            mmd_loss

    ):

        """
        Compute total SICDA loss
        """

        total_loss = (

            det_loss +

            self.lambda_ssl * ssl_loss +

            self.lambda_adv * adv_loss +

            self.lambda_cm * cm_loss +

            self.lambda_mmd * mmd_loss

        )

        loss_dict = {

            "total_loss": total_loss,

            "det_loss": det_loss,

            "ssl_loss": ssl_loss,

            "adv_loss": adv_loss,

            "cm_loss": cm_loss,

            "mmd_loss": mmd_loss

        }

        return total_loss, loss_dict


# ==============================================================
# Self-test
# ==============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TotalLoss Self-Test")
    print("=" * 70)

    try:

        from config import Config

        criterion = TotalLoss(

            lambda_ssl=Config.LAMBDA_SSL,

            lambda_adv=Config.LAMBDA_ADV,

            lambda_cm=Config.LAMBDA_CM,

            lambda_mmd=Config.LAMBDA_MMD

        )

        print("Using Config lambdas")
        print(f"  λ_ssl = {Config.LAMBDA_SSL}")
        print(f"  λ_adv = {Config.LAMBDA_ADV}")
        print(f"  λ_cm  = {Config.LAMBDA_CM}")
        print(f"  λ_mmd = {Config.LAMBDA_MMD}")

    except Exception:

        criterion = TotalLoss(

            lambda_ssl=1.0,

            lambda_adv=0.5,

            lambda_cm=0.2,

            lambda_mmd=0.1

        )

        print("Using default lambdas")
        print("  λ_ssl = 1.0")
        print("  λ_adv = 0.5")
        print("  λ_cm  = 0.2")
        print("  λ_mmd = 0.1")

    det_loss = torch.tensor(1.0, requires_grad=True)
    ssl_loss = torch.tensor(0.5, requires_grad=True)
    adv_loss = torch.tensor(0.3, requires_grad=True)
    cm_loss = torch.tensor(0.2, requires_grad=True)
    mmd_loss = torch.tensor(0.1, requires_grad=True)

    total, loss_dict = criterion(

        det_loss=det_loss,

        ssl_loss=ssl_loss,

        adv_loss=adv_loss,

        cm_loss=cm_loss,

        mmd_loss=mmd_loss

    )

    print("\nIndividual losses:")

    for k, v in loss_dict.items():

        print(f"  {k:12s}: {float(v):.4f}")

    print("\nBackward check...")

    total.backward()

    print("  Grad OK (det_loss.grad =", float(det_loss.grad), ")")

    print("=" * 70)
    print("TotalLoss test completed successfully")
    print("=" * 70)