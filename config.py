import os
import torch


class Config:

    #############################################################
    # Project
    #############################################################

    PROJECT_NAME = "SICDA"

    SEED = 42

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    GPU_NAME = "RTX4090D"

    USE_AMP = True

    MULTI_GPU = False

    NUM_WORKERS = 8

    PIN_MEMORY = True


    #############################################################
    # Dataset
    #############################################################

    NUM_CLASSES = 4

    CLASS_NAMES = [
        "background",
        "person",
        "car",
        "bike"
    ]


    #############################################################
    # Source Dataset
    # COCO RGB
    #############################################################

    SOURCE_ROOT = (
        r"/home/tq_saeed/saeed project/datasets/FLIR_dataset/FLIR_ADAS_v2"
    )


    SOURCE_IMAGE_DIR = os.path.join(
        SOURCE_ROOT,
        "train2017"
    )


    SOURCE_ANNOTATION = os.path.join(
        SOURCE_ROOT,
        "images_rgb_train",
        "coco.json"
    )


    #############################################################
    # Target Dataset
    # FLIR ADAS v2 RGB + Thermal
    #############################################################

    TARGET_ROOT = (
        r"/home/tq_saeed/saeed project/datasets/coco"
    )


    TARGET_RGB_DIR = os.path.join(
        TARGET_ROOT,
        "images_rgb_train"
    )


    TARGET_IR_DIR = os.path.join(
        TARGET_ROOT,
        "train2017"
    )


    TARGET_ANNOTATION = os.path.join(
        TARGET_ROOT,
        "annotations_trainval2017",
        "annotations",
        "instances_train2017.json"
    )


    #############################################################
    # Image Settings
    #############################################################

    IMAGE_SIZE = 512

    MIN_SIZE = 512

    MAX_SIZE = 640


    PIXEL_MEAN = [
        0.485,
        0.456,
        0.406
    ]


    PIXEL_STD = [
        0.229,
        0.224,
        0.225
    ]


    #############################################################
    # Training
    #############################################################

    EPOCHS = 5

    BATCH_SIZE = 2


    LEARNING_RATE = 5e-5

    WEIGHT_DECAY = 1e-4

    MOMENTUM = 0.9



    #############################################################
    # Optimizer
    #############################################################

    OPTIMIZER = "AdamW"

    LR_SCHEDULER = "CosineAnnealingLR"

    MIN_LR = 1e-6



    #############################################################
    # Backbone
    #############################################################

    BACKBONE = "resnet50"

    PRETRAINED = True

    USE_FPN = True

    FEATURE_DIM = 256



    #############################################################
    # Cross Modal Attention
    #############################################################

    USE_CROSS_MODAL_ATTENTION = True

    ATTENTION_HEADS = 8

    ATTENTION_DIM = 256



    #############################################################
    # Illumination Module
    #############################################################

    USE_ILLUMINATION = True



    #############################################################
    # SSL
    #############################################################

    USE_SSL = True

    PROJECTION_DIM = 256

    TEMPERATURE = 0.5

    EMBEDDING_DIM = 128



    #############################################################
    # Cross Modal Consistency
    #############################################################

    USE_CM_LOSS = True



    #############################################################
    # Domain Adaptation
    #############################################################

    USE_GRL = True

    DOMAIN_CLASSES = 2

    DOMAIN_HIDDEN = 1024



    #############################################################
    # MMD
    #############################################################

    USE_MMD = True

    MMD_KERNEL = "rbf"

    MMD_SIGMA = 1.0



    #############################################################
    # Detector
    #############################################################

    DETECTOR = "FasterRCNN"


    # Important:
    # Keep low for better recall

    SCORE_THRESHOLD = 0.05

    NMS_THRESHOLD = 0.5



    #############################################################
    # Anchor Configuration
    #############################################################

    ANCHOR_SIZES = (
        (16,),
        (32,),
        (64,),
        (128,)
    )


    ANCHOR_ASPECT_RATIOS = (
        (0.5, 1.0, 2.0),
        (0.5, 1.0, 2.0),
        (0.5, 1.0, 2.0),
        (0.5, 1.0, 2.0)
    )



    #############################################################
    # Loss Weights
    #############################################################

    # Original stable SICDA setting

    LAMBDA_SSL = 0.25

    LAMBDA_ADV = 0.10

    LAMBDA_CM = 0.10

    LAMBDA_MMD = 0.05



    #############################################################
    # Logging
    #############################################################

    PRINT_FREQ = 20

    SAVE_FREQ = 5

    VALIDATE_FREQ = 1



    #############################################################
    # Output
    #############################################################

    CHECKPOINT_DIR = "./checkpoints"

    LOG_DIR = "./logs"

    OUTPUT_DIR = "./outputs"

    DETECTION_DIR = "./outputs/detection"

    TENSORBOARD_DIR = "./tensorboard"



    #############################################################
    # Ablation
    #############################################################

    ABLATION = {

        "attention": True,

        "illumination": True,

        "ssl": True,

        "domain": True,

        "mmd": True,

        "consistency": True
    }



#############################################################
# Create Directories
#############################################################

folders = [

    Config.CHECKPOINT_DIR,

    Config.LOG_DIR,

    Config.OUTPUT_DIR,

    Config.DETECTION_DIR,

    Config.TENSORBOARD_DIR

]


for folder in folders:

    os.makedirs(
        folder,
        exist_ok=True
    )