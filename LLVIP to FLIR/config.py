
# ================================================================
# SICDA CONFIGURATION
# LLVIP SOURCE  ->  FLIR TARGET
# STABILITY-OPTIMIZED VERSION
# ================================================================

import os
import torch


class Config:

    # ============================================================
    # PROJECT
    # ============================================================

    PROJECT_NAME = "SICDA_LLVIP_SOURCE_FLIR_TARGET"

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    SEED = 42


    # ============================================================
    # DATASET DOMAIN
    # ============================================================

    SOURCE_DOMAIN = "LLVIP"
    TARGET_DOMAIN = "FLIR"


    # ============================================================
    # CLASSES
    # ============================================================

    NUM_CLASSES = 4

    CLASS_NAMES = {
        0: "background",
        1: "person",
        2: "car",
        3: "bike",
    }

    ID_TO_CLASS = {
        0: "background",
        1: "person",
        2: "car",
        3: "bike",
    }

    CLASS_TO_ID = {
        "background": 0,
        "person": 1,
        "car": 2,
        "bike": 3,
        "bicycle": 3,
    }


    # ============================================================
    # LLVIP SOURCE DATASET
    # ============================================================

    LLVIP_ROOT = r"E:\saeedwork\LLVIP\LLVIP"

    SOURCE_VISIBLE_TRAIN_DIR = os.path.join(
        LLVIP_ROOT,
        "visible",
        "train"
    )

    SOURCE_INFRARED_TRAIN_DIR = os.path.join(
        LLVIP_ROOT,
        "infrared",
        "train"
    )

    SOURCE_ANNOTATION_DIR = os.path.join(
        LLVIP_ROOT,
        "Generated_Annotations",
        "train",
        "labels"
    )


    # ============================================================
    # LLVIP ALIASES
    # ============================================================

    SOURCE_RGB_TRAIN_DIR = SOURCE_VISIBLE_TRAIN_DIR

    SOURCE_VISIBLE_DIR = SOURCE_VISIBLE_TRAIN_DIR

    SOURCE_IR_TRAIN_DIR = SOURCE_INFRARED_TRAIN_DIR

    SOURCE_INFRARED_DIR = SOURCE_INFRARED_TRAIN_DIR

    SOURCE_IR_DIR = SOURCE_INFRARED_TRAIN_DIR

    SOURCE_LABEL_DIR = SOURCE_ANNOTATION_DIR

    SOURCE_ANNOTATION_TRAIN_DIR = SOURCE_ANNOTATION_DIR


    # ============================================================
    # LLVIP VALIDATION / TEST
    # ============================================================

    SOURCE_VISIBLE_VAL_DIR = os.path.join(
        LLVIP_ROOT,
        "visible",
        "test"
    )

    SOURCE_INFRARED_VAL_DIR = os.path.join(
        LLVIP_ROOT,
        "infrared",
        "test"
    )

    SOURCE_IR_VAL_DIR = SOURCE_INFRARED_VAL_DIR

    SOURCE_VISIBLE_TEST_DIR = SOURCE_VISIBLE_VAL_DIR

    SOURCE_INFRARED_TEST_DIR = SOURCE_INFRARED_VAL_DIR


    # ============================================================
    # FLIR TARGET DATASET
    # ============================================================

    FLIR_ROOT = r"E:\saeedwork\FLIR_dataset\FLIR_ADAS_v2"


    # ------------------------------------------------------------
    # FLIR TRAIN
    # ------------------------------------------------------------

    TARGET_RGB_TRAIN_DIR = os.path.join(
        FLIR_ROOT,
        "images_rgb_train",
        "data"
    )

    TARGET_INFRARED_TRAIN_DIR = os.path.join(
        FLIR_ROOT,
        "images_thermal_train",
        "data"
    )


    # ------------------------------------------------------------
    # FLIR VALIDATION
    # ------------------------------------------------------------

    TARGET_VAL_RGB_DIR = os.path.join(
        FLIR_ROOT,
        "images_rgb_val",
        "data"
    )

    TARGET_VAL_INFRARED_DIR = os.path.join(
        FLIR_ROOT,
        "images_thermal_val",
        "data"
    )


    # ============================================================
    # FLIR ALIASES
    # ============================================================

    TARGET_RGB_DIR = TARGET_RGB_TRAIN_DIR

    TARGET_VISIBLE_TRAIN_DIR = TARGET_RGB_TRAIN_DIR

    TARGET_VISIBLE_DIR = TARGET_RGB_TRAIN_DIR

    TARGET_IR_TRAIN_DIR = TARGET_INFRARED_TRAIN_DIR

    TARGET_INFRARED_DIR = TARGET_INFRARED_TRAIN_DIR

    TARGET_IR_DIR = TARGET_INFRARED_TRAIN_DIR

    TARGET_VAL_IR_DIR = TARGET_VAL_INFRARED_DIR

    TARGET_INFRARED_VAL_DIR = TARGET_VAL_INFRARED_DIR

    TARGET_IR_VAL_DIR = TARGET_VAL_INFRARED_DIR

    TARGET_VISIBLE_VAL_DIR = TARGET_VAL_RGB_DIR


    # ============================================================
    # FLIR ANNOTATIONS
    # ============================================================

    TARGET_VAL_ANNOTATION = os.path.join(
        FLIR_ROOT,
        "images_rgb_train",
        "coco.JSON"
    )

    TARGET_ANNOTATION = TARGET_VAL_ANNOTATION

    TARGET_COCO_ANNOTATION = TARGET_VAL_ANNOTATION


    # ============================================================
    # IMAGE SETTINGS
    # ============================================================

    IMAGE_SIZE = 512

    IMAGE_WIDTH = 512

    IMAGE_HEIGHT = 512

    RGB_CHANNELS = 3

    THERMAL_CHANNELS = 1


    # ============================================================
    # TRAINING
    # ============================================================

    EPOCHS = 30

    BATCH_SIZE = 1

    NUM_WORKERS = 2

    PIN_MEMORY = True

    DROP_LAST = False


    # ============================================================
    # OPTIMIZER
    # ============================================================
    #
    # IMPORTANT:
    # Reduced from 1e-4 to 2e-5 because the previous learning
    # rate caused feature values to become NaN/Inf during epoch 3.
    # ============================================================

    OPTIMIZER = "AdamW"

    LR = 2e-5

    MIN_LR = 1e-6

    WEIGHT_DECAY = 1e-4

    BETAS = (
        0.9,
        0.999
    )

    EPS = 1e-8


    # ============================================================
    # GRADIENT STABILITY
    # ============================================================

    USE_GRADIENT_CLIPPING = True

    GRAD_CLIP_NORM = 5.0

    GRAD_CLIP_VALUE = None


    # ============================================================
    # LR SCHEDULER
    # ============================================================

    LR_SCHEDULER = "CosineAnnealingLR"

    T_MAX = EPOCHS

    ETA_MIN = MIN_LR


    # ============================================================
    # MODEL
    # ============================================================

    BACKBONE = "resnet50"

    PRETRAINED_BACKBONE = True

    FPN_CHANNELS = 256

    FEATURE_DIM = 256


    # ============================================================
    # FPN
    # ============================================================

    FPN_LEVELS = [
        "p2",
        "p3",
        "p4",
        "p5",
    ]


    # ============================================================
    # CROSS MODAL ATTENTION
    # ============================================================

    ATTENTION_HEADS = 4

    WINDOW_SIZE = 7

    CROSS_MODAL_DIM = 256

    ATTENTION_DROPOUT = 0.1


    # ============================================================
    # PROJECTION HEAD
    # ============================================================

    PROJECTION_DIM = 256

    PROJECTION_HIDDEN_DIM = 512


    # ============================================================
    # SELF-SUPERVISED LEARNING
    # ============================================================

    SSL_TEMPERATURE = 0.07

    SSL_PROJECTION_DIM = 256

    SSL_HIDDEN_DIM = 512


    # ============================================================
    # DOMAIN ADAPTATION
    # ============================================================

    USE_DANN = True

    USE_MMD = True

    USE_GRL = True


    # ============================================================
    # GRL
    # ============================================================
    #
    # IMPORTANT:
    # Reduced because batch size is only 1.
    # Full GRL strength can destabilize the backbone.
    # ============================================================

    GRL_LAMBDA = 0.0

    GRL_MAX_LAMBDA = 0.1

    GRL_WARMUP_EPOCHS = 5


    # ============================================================
    # DOMAIN DISCRIMINATOR
    # ============================================================

    DOMAIN_FEATURE_DIM = 256

    DOMAIN_HIDDEN_DIM = 256

    DOMAIN_NUM_CLASSES = 2

    DOMAIN_DROPOUT = 0.1


    # ============================================================
    # MMD
    # ============================================================

    MMD_KERNEL = "rbf"

    # Safer values for numerical stability
    MMD_GAMMA = 0.1

    MMD_SIGMA = 1.0

    MMD_EPS = 1e-8


    # ============================================================
    # LOSS WEIGHTS
    # ============================================================
    #
    # IMPORTANT:
    # Previous configuration used all weights = 1.0.
    # With batch size 1, adaptation losses dominated
    # the detection loss and caused instability.
    # ============================================================

    LAMBDA_DETECTION = 1.0

    LAMBDA_SSL = 0.1

    LAMBDA_DANN = 0.1

    LAMBDA_MMD = 0.05

    LAMBDA_CONTRASTIVE = 0.1


    # ============================================================
    # LOSS STABILITY
    # ============================================================

    LOSS_EPS = 1e-8

    MAX_LOSS_VALUE = 100.0

    SKIP_NAN_BATCHES = True

    SKIP_INF_BATCHES = True


    # ============================================================
    # DETECTOR
    # ============================================================

    DETECTOR = "FasterRCNN"

    DETECTION_SCORE_THRESHOLD = 0.05

    NMS_THRESHOLD = 0.50

    IOU_THRESHOLD = 0.50


    # ============================================================
    # EVALUATION
    # ============================================================

    EVAL_IOU_THRESHOLD = 0.50

    MAP_IOU_THRESHOLD = 0.50

    MAP50_THRESHOLD = 0.50

    MAP75_THRESHOLD = 0.75


    # ============================================================
    # PSEUDO LABELING
    # ============================================================

    USE_PSEUDO_LABELS = False

    PSEUDO_LABEL_THRESHOLD = 0.70

    PSEUDO_LABEL_NMS_THRESHOLD = 0.50


    # ============================================================
    # DATA AUGMENTATION
    # ============================================================

    USE_AUGMENTATION = True

    RANDOM_HORIZONTAL_FLIP = True

    RANDOM_VERTICAL_FLIP = False

    RANDOM_CROP = False

    RANDOM_RESIZE = True


    # ============================================================
    # NORMALIZATION
    # ============================================================

    RGB_MEAN = [
        0.485,
        0.456,
        0.406
    ]

    RGB_STD = [
        0.229,
        0.224,
        0.225
    ]

    THERMAL_MEAN = [
        0.5
    ]

    THERMAL_STD = [
        0.5
    ]

    # Backward compatibility aliases
    PIXEL_MEAN = RGB_MEAN

    PIXEL_STD = RGB_STD


    # ============================================================
    # CHECKPOINTS
    # ============================================================

    CHECKPOINT_DIR = os.path.join(
        "checkpoints"
    )

    BEST_MODEL_PATH = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )

    LAST_MODEL_PATH = os.path.join(
        CHECKPOINT_DIR,
        "last_model.pth"
    )


    # ============================================================
    # LOGGING
    # ============================================================

    LOG_DIR = os.path.join(
        "logs"
    )

    RESULTS_DIR = os.path.join(
        "results"
    )

    EVALUATION_DIR = os.path.join(
        RESULTS_DIR,
        "evaluation"
    )

    ABLATION_DIR = os.path.join(
        RESULTS_DIR,
        "ablation"
    )

    SENSITIVITY_DIR = os.path.join(
        RESULTS_DIR,
        "sensitivity"
    )

    COMPLEXITY_DIR = os.path.join(
        RESULTS_DIR,
        "complexity"
    )


    # ============================================================
    # REPRODUCIBILITY
    # ============================================================

    DETERMINISTIC = False

    BENCHMARK = True


    # ============================================================
    # DEBUG
    # ============================================================

    DEBUG = False

    DEBUG_SAMPLES = 10

    DETECT_ANOMALY = False


    # ============================================================
    # PRINT CONFIGURATION
    # ============================================================

    @classmethod
    def print_config(cls):

        print()
        print("=" * 78)
        print("             SICDA CONFIGURATION - STABILITY OPTIMIZED")
        print("=" * 78)

        print()

        print("DEVICE")
        print("-" * 78)
        print(f"Device              : {cls.DEVICE}")

        if torch.cuda.is_available():
            print(
                f"GPU                 : "
                f"{torch.cuda.get_device_name(0)}"
            )

        print()

        print("DOMAINS")
        print("-" * 78)
        print(f"Source Domain       : {cls.SOURCE_DOMAIN}")
        print(f"Target Domain       : {cls.TARGET_DOMAIN}")

        print()

        print("CLASSES")
        print("-" * 78)
        print(f"NUM_CLASSES         : {cls.NUM_CLASSES}")
        print(f"CLASS_NAMES         : {cls.CLASS_NAMES}")

        print()

        print("TRAINING")
        print("-" * 78)
        print(f"Epochs              : {cls.EPOCHS}")
        print(f"Batch size          : {cls.BATCH_SIZE}")
        print(f"Learning rate       : {cls.LR}")
        print(f"Weight decay        : {cls.WEIGHT_DECAY}")
        print(f"Optimizer           : {cls.OPTIMIZER}")
        print(f"Gradient clipping   : {cls.USE_GRADIENT_CLIPPING}")
        print(f"Clip norm           : {cls.GRAD_CLIP_NORM}")

        print()

        print("DOMAIN ADAPTATION")
        print("-" * 78)
        print(f"DANN                : {cls.USE_DANN}")
        print(f"MMD                 : {cls.USE_MMD}")
        print(f"GRL                 : {cls.USE_GRL}")
        print(f"GRL max lambda      : {cls.GRL_MAX_LAMBDA}")
        print(f"GRL warmup epochs   : {cls.GRL_WARMUP_EPOCHS}")

        print()

        print("LOSS WEIGHTS")
        print("-" * 78)
        print(f"Detection           : {cls.LAMBDA_DETECTION}")
        print(f"SSL                 : {cls.LAMBDA_SSL}")
        print(f"DANN                : {cls.LAMBDA_DANN}")
        print(f"MMD                 : {cls.LAMBDA_MMD}")
        print(f"Contrastive         : {cls.LAMBDA_CONTRASTIVE}")

        print()

        print("NUMERICAL STABILITY")
        print("-" * 78)
        print(f"Loss epsilon        : {cls.LOSS_EPS}")
        print(f"MMD epsilon         : {cls.MMD_EPS}")
        print(f"Skip NaN batches    : {cls.SKIP_NAN_BATCHES}")
        print(f"Skip Inf batches    : {cls.SKIP_INF_BATCHES}")

        print()
        print("=" * 78)


# ====================================================================
# CONFIG TEST
# ====================================================================

if __name__ == "__main__":
    Config.print_config()

