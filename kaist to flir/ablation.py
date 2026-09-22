"""
==============================================================
File : ablation.py

SICDA Ablation Study Module


Experiments:

A0:
    Detection Only


A1:
    Detection + Contrastive SSL


A2:
    Detection + SSL + Domain Adaptation


A3:
    Detection + SSL + DANN + MMD


A4:
    Full SICDA

    Detection
    +
    Contrastive
    +
    DANN
    +
    MMD
    +
    Illumination Fusion



Author:
Muhammad Saeed

==============================================================
"""


import torch



# ============================================================
# Ablation Configuration
# ============================================================


ABLATION_CONFIGS = {


    "Baseline":{


        "use_ssl":False,

        "use_adv":False,

        "use_mmd":False,

        "use_illumination":False,

        "description":

        "Detection loss only"

    },



    "SSL":{


        "use_ssl":True,

        "use_adv":False,

        "use_mmd":False,

        "use_illumination":False,


        "description":

        "Detection + Contrastive Loss"

    },



    "SSL_DANN":{


        "use_ssl":True,

        "use_adv":True,

        "use_mmd":False,

        "use_illumination":False,


        "description":

        "Detection + SSL + Domain Adversarial Training"

    },



    "SSL_DANN_MMD":{


        "use_ssl":True,

        "use_adv":True,

        "use_mmd":True,

        "use_illumination":False,


        "description":

        "Detection + SSL + DANN + MMD"

    },



    "Full_SICDA":{


        "use_ssl":True,

        "use_adv":True,

        "use_mmd":True,

        "use_illumination":True,


        "description":

        "Complete SICDA Framework"

    }

}




# ============================================================
# Select Ablation
# ============================================================


def get_ablation_config(name):


    if name not in ABLATION_CONFIGS:


        raise ValueError(

            f"Unknown Ablation: {name}"

        )


    return ABLATION_CONFIGS[name]




# ============================================================
# Compute Total Loss According
# to Ablation Setting
# ============================================================


def compute_ablation_loss(

        det_loss,

        ssl_loss,

        adv_loss,

        mmd_loss,

        cm_loss,

        config,

        lambda_ssl=1.0,

        lambda_adv=0.5,

        lambda_mmd=0.1,

        lambda_cm=0.2

):



    loss=det_loss



    loss_dict={

        "det_loss":det_loss

    }




    # ---------------- SSL ----------------


    if config["use_ssl"]:


        loss += lambda_ssl * ssl_loss


        loss_dict["ssl_loss"]=ssl_loss



    else:


        loss_dict["ssl_loss"]=torch.tensor(

            0.0,

            device=det_loss.device

        )





    # ---------------- DANN ----------------


    if config["use_adv"]:


        loss += lambda_adv * adv_loss


        loss_dict["adv_loss"]=adv_loss


    else:


        loss_dict["adv_loss"]=torch.tensor(

            0.0,

            device=det_loss.device

        )





    # ---------------- MMD ----------------


    if config["use_mmd"]:


        loss += lambda_mmd * mmd_loss


        loss_dict["mmd_loss"]=mmd_loss



    else:


        loss_dict["mmd_loss"]=torch.tensor(

            0.0,

            device=det_loss.device

        )





    # ---------------- Illumination Fusion ----------------


    if config["use_illumination"]:


        loss += lambda_cm * cm_loss


        loss_dict["cm_loss"]=cm_loss



    else:


        loss_dict["cm_loss"]=torch.tensor(

            0.0,

            device=det_loss.device

        )




    loss_dict["total_loss"]=loss



    return loss,loss_dict





# ============================================================
# Print Ablation Table
# ============================================================


def print_ablation_table():



    print()

    print("="*80)

    print("SICDA Ablation Study Configurations")

    print("="*80)



    for name,cfg in ABLATION_CONFIGS.items():


        print()

        print(name)


        print("-"*50)


        print(

            cfg["description"]

        )


        print(

            "SSL:",

            cfg["use_ssl"],

            "| ADV:",

            cfg["use_adv"],

            "| MMD:",

            cfg["use_mmd"],

            "| Illumination:",

            cfg["use_illumination"]

        )


    print()

    print("="*80)




# ============================================================
# Test
# ============================================================


if __name__=="__main__":



    print_ablation_table()



    device="cuda" if torch.cuda.is_available() else "cpu"



    det=torch.tensor(

        1.0,

        device=device,

        requires_grad=True

    )


    ssl=torch.tensor(

        0.5,

        device=device

    )


    adv=torch.tensor(

        0.3,

        device=device

    )


    mmd=torch.tensor(

        0.2,

        device=device

    )


    cm=torch.tensor(

        0.1,

        device=device

    )



    config=get_ablation_config(

        "Full_SICDA"

    )



    loss,loss_dict=compute_ablation_loss(

        det,

        ssl,

        adv,

        mmd,

        cm,

        config

    )



    print()

    print(

        "Total Loss:",

        loss.item()

    )


    print()

    print(

        "Ablation Test Passed"

    )