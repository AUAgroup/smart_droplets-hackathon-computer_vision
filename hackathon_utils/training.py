import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F
from hackathon_utils.training_config import *
from hackathon_utils.evaluation import evaluate
from hackathon_utils.visualization import visualize_segmentation

logger = logging.getLogger(__name__)
#logger.addHandler(logging.NullHandler())

def train_model(
    model,
    train_loader,
    val_loader,
    max_lr=1e-4,
    weight_decay=1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=weight_decay
    )

    best_weights = None
    best_val_loss = np.inf
    best_avg_jaccard_scores = -np.inf  # fixed typo & safer init for "maximize" metric

    patience = PATIENCE
    best_epoch = 1
    losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # [B, C, H, W]

            loss = F.cross_entropy(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # accumulate epoch loss (sample-weighted)
            running_loss += loss.item() * images.size(0)

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"[Epoch {epoch}] Batch {batch_idx+1}/{len(train_loader)} — "
                    f"Loss: {loss.item():.4f}"
                )

        # epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(round(epoch_loss, 4))
        logger.info(f"==> Epoch {epoch} Complete: Training Avg. Loss = {epoch_loss:.4f}")

        # ---- evaluation ----
        model.eval()
        logger.info(f"[Epoch {epoch}] Evaluating on the training set")
        train_loss, train_avg_jaccard_scores, train_avg_jaccard_per_class = evaluate(
            model, train_loader
        )

        logger.info(f"[Epoch {epoch}] Evaluating on the validation set")
        val_loss, val_avg_jaccard_scores, val_avg_jaccard_per_class = evaluate(
            model, val_loader
        )

        logger.info(
            f"[Epoch {epoch}] "
            f"val_loss={val_loss:.4f}, "
            f"train_mIoU={train_avg_jaccard_scores:.4f}, "
            f"val_mIoU={val_avg_jaccard_scores:.4f}, "
            f"train_per_class={train_avg_jaccard_per_class}, "
            f"val_per_class={val_avg_jaccard_per_class}"
        )

        visualize_segmentation(
            model,
            val_loader,
            epoch
        )

        # ---- early stopping condition (maximize mIoU; tie-break with lower val loss) ----
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
            best_avg_jaccard_scores = val_avg_jaccard_scores
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience = PATIENCE
            best_epoch = epoch
        else:
            patience -= 1
            logger.info(f"[Epoch {epoch}] No improvement. Patience: {patience}")
            if patience == 0:
                logger.info("[Early Stopping] Restoring best weights and finalizing.")
                if best_weights is not None:
                    model.load_state_dict(best_weights)

    # after all epochs, ensure best weights are used (if any saved)
    if best_weights is not None:
        model.load_state_dict(best_weights)