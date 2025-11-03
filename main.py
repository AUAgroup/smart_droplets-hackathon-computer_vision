from hackathon_utils.data_config import *
from hackathon_utils.training_config import *

from hackathon_utils.reproducibility import set_seed
from hackathon_utils.auxiliar import create_folder_if_not_exists, count_params
from hackathon_utils.datasets import get_dataloaders
import torch

from hackathon_utils.training import train_model
from hackathon_utils.presentation import segment_images, evaluate_iou
import segmentation_models_pytorch as smp
import timm

from hackathon_utils.log import set_log
import logging
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_config = set_log()
logging.getLogger().handlers.clear()

logging.config.dictConfig(log_config)
logger = logging.getLogger(__name__) # __name__ gives a hierarchical logger name

# Reproducibility across Python, NumPy, PyTorch, CUDA (as implemented in your helper)
set_seed()

# Ensure the output folder exists before any file I/O happens later (e.g., CSV/logs, predictions)
create_folder_if_not_exists(OUTPUT_FOLDER_NAME)

try:
    # -------------------------------
    # Data config / preprocessing
    # -------------------------------
    # We instantiate a timm backbone ONLY to derive its canonical data config
    # (input size, mean/std, interpolation, crop %, etc.). num_classes=0 makes it a feature extractor.
    logger.info("Loading backbone for data config...")
    timm_backbone_for_cfg = timm.create_model(
        BACKBONE_NAME,        # e.g., "convnextv2_nano", "resnet50", etc.
        num_classes=0,        # no classifier head; we only need preprocessing params
        pretrained=True       # load ImageNet weights to get the right normalization defaults
    )

    # Resolve the model-specific data config (dict with keys like 'input_size', 'mean', 'std', ...)
    data_config = timm.data.resolve_model_data_config(timm_backbone_for_cfg)
    logger.info(f"Data config: {data_config}")

    # Build train/test transforms consistent with the backbone’s expectations.
    # Note: These include normalization with backbone-specific mean/std.
    preprocessing_train = timm.data.create_transform(
        **data_config, is_training=True
    )
    preprocessing_test = timm.data.create_transform(
        **data_config, is_training=False
    )

    # -------------------------------
    # Dataloaders
    # -------------------------------
    # Create train/val dataloaders using the resolved data_config so that
    # images are transformed correctly (size, normalization, etc.).
    train_loader, val_loader = get_dataloaders(data_config)

    # Helpful sanity check: confirm dataset sizes after any filtering/splitting.
    logger.info(
        f"{len(train_loader.dataset)}, {len(val_loader.dataset)}"
    )

    # -------------------------------
    # Model build + training
    # -------------------------------
    try:
        logger.info("Creating segmentation model...")

        # Build a segmentation model with segmentation_models_pytorch (smp).
        # - ARCHITECTURE_NAME: e.g., "Unet", "FPN", "DeepLabV3", etc.
        # - encoder_name expects a timm backbone id. Here it’s prefixed "tu-"
        #   which is TIMM-Unicode (a.k.a. timm’s new registry) notation for many encoders.
        #   Make sure your BACKBONE_NAME exists under that prefix in your smp/timm versions.
        model = smp.create_model(
            ARCHITECTURE_NAME,
            encoder_name=f"tu-{BACKBONE_NAME}",  # e.g., "tu-convnextv2_nano"
            in_channels=3,                        # RGB inputs
            encoder_weights="imagenet",           # initialize encoder with ImageNet weights
            classes=NUM_CLASSES,                  # number of segmentation classes
        ).to(device)                              # move to CPU/GPU per your 'device'

        # Log model size for transparency and capacity checks
        total_params, trainable_params = count_params(model)
        logger.info(
            f"Params — total: {total_params/1e6:.3f}M | "
            f"trainable: {trainable_params/1e6:.3f}M"
        )

        # Kick off training with your helper. Assumes:
        # - train_model handles epochs, validation, checkpointing/early stopping (if any),
        #   and logs losses/metrics internally.
        # - LEARNING_RATE and WEIGHT_DECAY are defined globals/hyperparams.
        train_model(
            model,
            train_loader,
            val_loader,
            max_lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        # After training, run inference/segmentation on a test image folder and save outputs.
        # Assumes:
        # - segment_images iterates files in TEST_IMG_DIR, writes masks/overlays to RESULTS_IMG_DIR,
        #   and uses the same device as training.
        segment_images(
            model,
            input_folder=TEST_IMG_DIR,
            output_folder=RESULTS_IMG_DIR,
            device=device
        )

    except Exception as e:
        # Any failure inside the training block (model creation, training loop, or post-processing)
        # will be logged here with full traceback for easier debugging.
        logger.error(f"Training block failed: {e}")
        logger.error(traceback.format_exc())

except Exception as e:
    # Catch-all to avoid silent crashes and preserve a full stack trace for post-mortem analysis.
    logger.error("Unexpected error at top-level try/with block.")
    logger.error(traceback.format_exc())