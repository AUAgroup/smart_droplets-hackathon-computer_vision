import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F
from hackathon_utils.training_config import *
from hackathon_utils.evaluation import evaluate
from hackathon_utils.visualization import visualize_segmentation

logger = logging.getLogger(__name__)
#logger.addHandler(logging.NullHandler())  # optional: prevents "no handler" warnings if no global logging setup

# ---------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    max_lr=1e-4,
    weight_decay=1e-4,
):
    """
    Train and evaluate a segmentation model with early stopping and visualization.

    Args:
        model: PyTorch segmentation model.
        train_loader: Dataloader for training set.
        val_loader: Dataloader for validation set.
        max_lr: Learning rate for optimizer.
        weight_decay: L2 regularization term for optimizer.

    Returns:
        The model trained with best weights (based on validation IoU).
    """
    # Select device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize optimizer (AdamW recommended for segmentation)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=weight_decay
    )

    # Initialize best metrics and model copy for checkpointing
    best_weights = None
    best_val_loss = np.inf
    best_avg_jaccard_scores = -np.inf  # start at -inf because IoU is maximized

    # Initialize early stopping parameters
    patience = PATIENCE      # from training_config
    best_epoch = 1
    losses = []              # track training loss across epochs

    # -----------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        # -------------------------------------------------------------
        # Mini-batch training loop
        # -------------------------------------------------------------
        for batch_idx, (images, masks) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)  # expected shape: [B, C, H, W]

            # Compute cross-entropy loss (multi-class segmentation)
            loss = F.cross_entropy(outputs, masks)

            # Backpropagate loss
            loss.backward()

            # Optional: clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Parameter update
            optimizer.step()

            # Accumulate loss weighted by batch size
            running_loss += loss.item() * images.size(0)

            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"[Epoch {epoch}] Batch {batch_idx+1}/{len(train_loader)} — "
                    f"Loss: {loss.item():.4f}"
                )

        # -------------------------------------------------------------
        # End-of-epoch processing
        # -------------------------------------------------------------
        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(round(epoch_loss, 4))
        logger.info(f"==> Epoch {epoch} Complete: Training Avg. Loss = {epoch_loss:.4f}")

        # -------------------------------------------------------------
        # Evaluate model on training and validation sets
        # -------------------------------------------------------------
        model.eval()
        logger.info(f"[Epoch {epoch}] Evaluating on the training set")
        train_loss, train_avg_jaccard_scores, train_avg_jaccard_per_class = evaluate(
            model, train_loader
        )

        logger.info(f"[Epoch {epoch}] Evaluating on the validation set")
        val_loss, val_avg_jaccard_scores, val_avg_jaccard_per_class = evaluate(
            model, val_loader
        )

        # Log detailed metrics (loss + IoU)
        logger.info(
            f"[Epoch {epoch}] "
            f"val_loss={val_loss:.4f}, "
            f"train_mIoU={train_avg_jaccard_scores:.4f}, "
            f"val_mIoU={val_avg_jaccard_scores:.4f}, "
            f"train_per_class={train_avg_jaccard_per_class}, "
            f"val_per_class={val_avg_jaccard_per_class}"
        )

        # -------------------------------------------------------------
        # Visualization of predictions for qualitative inspection
        # -------------------------------------------------------------
        visualize_segmentation(
            model,
            val_loader,
            epoch
        )

        # -------------------------------------------------------------
        # Early stopping / checkpoint logic
        # -------------------------------------------------------------
        # Condition: model improves if validation IoU increases, or IoU ties but val_loss decreases
        improved = (
            (val_avg_jaccard_scores > best_avg_jaccard_scores) or
            (np.isclose(val_avg_jaccard_scores, best_avg_jaccard_scores) and val_loss < best_val_loss)
        )

        if improved:
            logger.info(
                f"[Epoch {epoch}] Saving Weights... "
                f"(mIoU: {best_avg_jaccard_scores:.4f} → {val_avg_jaccard_scores:.4f}, "
                f"val_loss: {best_val_loss:.4f} → {val_loss:.4f})"
            )
            # Save best model state
            best_avg_jaccard_scores = val_avg_jaccard_scores
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience = PATIENCE     # reset patience counter
            best_epoch = epoch
        else:
            # No improvement → decrease patience
            patience -= 1
            logger.info(f"[Epoch {epoch}] No improvement. Patience: {patience}")

            # Stop if patience runs out
            if patience == 0:
                logger.info("[Early Stopping] Restoring best weights and finalizing.")
                if best_weights is not None:
                    model.load_state_dict(best_weights)

    # -----------------------------------------------------------------
    # Final model restoration (best checkpoint)
    # -----------------------------------------------------------------
    if best_weights is not None:
        model.load_state_dict(best_weights)