"""
==============================================================
File : ablation_runner.py

SICDA Ablation Study Runner

Experiments:
1. Baseline FasterRCNN
2. + Cross Modal Attention
3. + Illumination Module
4. + SSL
5. + GRL Domain Adaptation
6. + MMD
7. Full SICDA

Output:
    outputs/ablation_results.csv

==============================================================
"""


import os
import csv
import torch
import subprocess
from datetime import datetime


from config import Config



###############################################################
# Ablation Experiments
###############################################################


ABLATION_EXPERIMENTS = {


    "Baseline": {

        "attention": False,

        "illumination": False,

        "ssl": False,

        "domain": False,

        "mmd": False,

        "consistency": False
    },


    "Attention": {

        "attention": True,

        "illumination": False,

        "ssl": False,

        "domain": False,

        "mmd": False,

        "consistency": False
    },


    "Attention+Illumination": {

        "attention": True,

        "illumination": True,

        "ssl": False,

        "domain": False,

        "mmd": False,

        "consistency": False
    },


    "Attention+Illumination+SSL": {

        "attention": True,

        "illumination": True,

        "ssl": True,

        "domain": False,

        "mmd": False,

        "consistency": False
    },


    "Attention+SSL+GRL": {

        "attention": True,

        "illumination": True,

        "ssl": True,

        "domain": True,

        "mmd": False,

        "consistency": False
    },


    "Attention+SSL+GRL+MMD": {

        "attention": True,

        "illumination": True,

        "ssl": True,

        "domain": True,

        "mmd": True,

        "consistency": False
    },


    "Full SICDA": {

        "attention": True,

        "illumination": True,

        "ssl": True,

        "domain": True,

        "mmd": True,

        "consistency": True
    }

}



###############################################################
# Output CSV
###############################################################


RESULT_FILE = (

    "./outputs/ablation_results.csv"

)



###############################################################
# Save Result
###############################################################


def save_result(
    name,
    metrics
):


    os.makedirs(

        "./outputs",

        exist_ok=True

    )


    file_exists = os.path.isfile(

        RESULT_FILE

    )


    with open(

        RESULT_FILE,

        "a",

        newline=""

    ) as f:


        writer = csv.writer(f)


        if not file_exists:


            writer.writerow([

                "Experiment",

                "mAP50",

                "mAP75",

                "mAP50_95",

                "Precision",

                "Recall",

                "F1"

            ])


        writer.writerow([

            name,

            metrics["mAP50"],

            metrics["mAP75"],

            metrics["mAP50_95"],

            metrics["Precision"],

            metrics["Recall"],

            metrics["F1"]

        ])



###############################################################
# Update Config Dynamically
###############################################################


def apply_ablation(setting):


    Config.ABLATION = setting


    Config.USE_CROSS_MODAL_ATTENTION = (

        setting["attention"]

    )


    Config.USE_ILLUMINATION = (

        setting["illumination"]

    )


    Config.USE_SSL = (

        setting["ssl"]

    )


    Config.USE_GRL = (

        setting["domain"]

    )


    Config.USE_MMD = (

        setting["mmd"]

    )


    Config.USE_CM_LOSS = (

        setting["consistency"]

    )



###############################################################
# Run One Experiment
###############################################################


def run_experiment(name, setting):


    print()

    print("="*70)

    print(
        "Running:",
        name
    )

    print("="*70)



    apply_ablation(setting)



    ###########################################################
    # Call Training
    ###########################################################

    command = [

        "python",

        "train.py"

    ]


    subprocess.run(

        command

    )



    ###########################################################
    # Evaluation
    ###########################################################

    from evaluator import evaluate


    metrics = evaluate()



    save_result(

        name,

        metrics

    )


    print(

        "Saved:",
        name

    )



###############################################################
# Main
###############################################################


if __name__ == "__main__":


    print()

    print("="*70)

    print("SICDA Ablation Study")

    print("="*70)



    for name, setting in ABLATION_EXPERIMENTS.items():


        run_experiment(

            name,

            setting

        )



    print()

    print("="*70)

    print(

        "Ablation Study Completed"

    )

    print(

        "Results saved:",
        RESULT_FILE

    )

    print("="*70)