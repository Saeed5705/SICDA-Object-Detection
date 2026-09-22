"""
==============================================================
File : config.py

SICDA Configuration

Source Domain:
    KAIST RGB-Thermal

Target Domain:
    FLIR RGB-Thermal

==============================================================
"""


import os


class Config:


    # ========================================================
    # Dataset Paths
    # ========================================================


    KAIST_ROOT = (
        "E:\saeedwork\Kaist"
    )


    FLIR_ROOT = (
        r"E:\saeedwork\FLIR_dataset\FLIR_ADAS_v2"
    )



    # ========================================================
    # Classes
    # ========================================================


    NUM_CLASSES = 4
    """
    0 Background
    1 Person
    2 Car
    3 Bicycle
    """



    CLASS_NAMES = {

        0: "Background",

        1: "Person",

        2: "Car",

        3: "Bicycle"

    }



    # ========================================================
    # Image Settings
    # ========================================================


    IMAGE_SIZE = 512



    # ========================================================
    # Training
    # ========================================================


    EPOCHS = 10


    BATCH_SIZE = 1


    NUM_WORKERS = 2



    # ========================================================
    # Optimizer
    # ========================================================


    OPTIMIZER = "AdamW"


    LR = 1e-4


    MIN_LR = 1e-6


    WEIGHT_DECAY = 1e-4



    # ========================================================
    # Feature Configuration
    # ========================================================


    FEATURE_DIM = 256



    # ========================================================
    # Cross Modal Attention
    # ========================================================


    ATTENTION_HEADS = 4


    WINDOW_SIZE = 7



    # ========================================================
    # Projection Head
    # ========================================================


    PROJECTION_DIM = 128


    TEMPERATURE = 0.07



    # ========================================================
    # Loss Weights
    # ========================================================


    LAMBDA_SSL = 0.1


    LAMBDA_ADV = 0.1


    LAMBDA_CM = 0.1


    LAMBDA_MMD = 0.05



    # ========================================================
    # Domain Discriminator
    # ========================================================


    DOMAIN_HIDDEN = 256



    # ========================================================
    # Evaluation
    # ========================================================


    IOU_THRESHOLD_50 = 0.50


    IOU_THRESHOLD_95 = 0.95



    # ========================================================
    # Checkpoints
    # ========================================================


    CHECKPOINT_DIR = (

        "./checkpoints"

    )


    RESULT_DIR = (

        "./results"

    )



    # ========================================================
    # Create Directories
    # ========================================================


    @staticmethod
    def create_dirs():

        os.makedirs(

            Config.CHECKPOINT_DIR,

            exist_ok=True

        )


        os.makedirs(

            Config.RESULT_DIR,

            exist_ok=True

        )



if __name__ == "__main__":


    Config.create_dirs()


    print("="*60)

    print("SICDA Configuration")

    print("="*60)

    print("Source :", Config.KAIST_ROOT)

    print("Target :", Config.FLIR_ROOT)

    print("Classes:", Config.NUM_CLASSES)

    print("Feature:", Config.FEATURE_DIM)

    print("="*60)