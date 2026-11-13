#!/usr/bin/env python3

import argparse
from pathlib import Path
import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import sys

try:
    from dataset import JunctionDataset
    from model import JunctionUNet
except ImportError:
    try:
        from .dataset import JunctionDataset
        from .model import JunctionUNet
    except ImportError:
        sys.path.append(str(Path(__file__).resolve().parent))
        from dataset import JunctionDataset
        from model import JunctionUNet


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 2e-3
DEFAULT_OUTPUT_DIR = Path("training_output")
DEFAULT_SIGMA = 2.0


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class PenaltyReducedFocalLoss(nn.Module):
    """
    CenterNet penalty-reduced focal loss for keypoint heatmaps.
    Down-weights penalties in the Gaussian neighborhood of ground-truth peaks,
    completely eliminating false-positive "blooming" haze.
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        pred = torch.clamp(pred, min=1e-5, max=1.0 - 1e-5)

        pos_mask = (target == 1.0)
        neg_mask = (target < 1.0)

        pos_loss = -torch.pow(1.0 - pred, self.alpha) * torch.log(pred)
        neg_loss = (
            -torch.pow(1.0 - target, self.beta)
            * torch.pow(pred, self.alpha)
            * torch.log(1.0 - pred)
        )

        num_pos = pos_mask.float().sum()
        loss = (pos_loss * pos_mask.float()).sum() + (neg_loss * neg_mask.float()).sum()
        return loss / torch.clamp(num_pos, min=1.0)


class WeightedHeatmapMSELoss(nn.Module):
    """
    Weighted MSE loss emphasizing junction peak centers.
    """

    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        weight = 1.0 + (self.pos_weight - 1.0) * target
        return torch.mean(weight * torch.pow(pred - target, 2))


# Backward compatibility alias
IntersectionLoss = WeightedHeatmapMSELoss


# ============================================================
# PEAK LOCALIZATION & EVALUATION
# ============================================================

def extract_peaks_from_heatmap(prob_map, threshold=0.25, kernel_size=3):
    """
    Extract discrete local maximum coordinates (x, y) from probability heatmap.
    """
    tensor = torch.from_numpy(prob_map).unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    max_pooled = F.max_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=pad)
    peaks_mask = (tensor == max_pooled) & (tensor > threshold)
    peaks_np = peaks_mask[0, 0].cpu().numpy()

    ys, xs = np.where(peaks_np)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack([xs, ys]).astype(np.int32)


def evaluate_junction_metrics(pred_coords, gt_coords, tolerance=5.0):
    """
    Compute Precision, Recall, and F1 score against ground truth coordinates.
    """
    if len(gt_coords) == 0 and len(pred_coords) == 0:
        return 1.0, 1.0, 1.0
    if len(gt_coords) == 0:
        return 0.0, 1.0, 0.0
    if len(pred_coords) == 0:
        return 1.0, 0.0, 0.0

    tree_gt = cKDTree(gt_coords)
    tree_pred = cKDTree(pred_coords)

    dists_pred, _ = tree_gt.query(pred_coords)
    tp_pred = np.sum(dists_pred <= tolerance)
    precision = float(tp_pred) / len(pred_coords)

    dists_gt, _ = tree_pred.query(gt_coords)
    tp_gt = np.sum(dists_gt <= tolerance)
    recall = float(tp_gt) / len(gt_coords)

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


# ============================================================
# VALIDATION LOOP
# ============================================================

@torch.no_grad()
def validate(model, val_loader, criterion, device, tolerance=5.0):
    model.eval()
    total_loss = 0.0
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for images, targets, paths in val_loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)

        preds = torch.sigmoid(logits)

        # Evaluate metrics per sample
        for i in range(images.size(0)):
            prob_map = preds[i, 0].cpu().numpy()
            target_map = targets[i, 0].cpu().numpy()

            pred_peaks = extract_peaks_from_heatmap(prob_map, threshold=0.25)
            gt_peaks = extract_peaks_from_heatmap(target_map, threshold=0.5)

            p, r, f = evaluate_junction_metrics(pred_peaks, gt_peaks, tolerance=tolerance)
            all_precisions.append(p)
            all_recalls.append(r)
            all_f1s.append(f)

    mean_loss = total_loss / len(val_loader.dataset)
    mean_p = float(np.mean(all_precisions)) if all_precisions else 0.0
    mean_r = float(np.mean(all_recalls)) if all_recalls else 0.0
    mean_f1 = float(np.mean(all_f1s)) if all_f1s else 0.0

    return mean_loss, mean_p, mean_r, mean_f1


# ============================================================
# TRAINING SCRIPT
# ============================================================

def train(
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    lr=DEFAULT_LR,
    output_dir=DEFAULT_OUTPUT_DIR,
    loss_type="focal",
    max_samples=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # Load dataset
    full_dataset = JunctionDataset(
        seeds_dir="data/seeds",
        image_dir="data/rotate",
        bw_dir="data/blackwhite_rotate",
        sigma=DEFAULT_SIGMA,
        augment=True,
        crop_size=(256, 256),
        max_samples=max_samples,
    )

    total_len = len(full_dataset)
    val_len = max(1, int(total_len * 0.2))
    train_len = total_len - val_len

    torch.manual_seed(100)
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len])
    # Disable augmentation on validation subset
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    print(f"Train samples: {train_len}, Validation samples: {val_len}")

    # Initialize model
    model = JunctionUNet(in_channels=1, out_channels=1, base_channels=16).to(device)

    # Loss & Optimizer
    if loss_type == "focal":
        criterion = PenaltyReducedFocalLoss()
    else:
        criterion = WeightedHeatmapMSELoss(pos_weight=10.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history_csv = output_dir / "training_history.csv"
    with open(history_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_precision", "val_recall", "val_f1", "lr"])

    best_f1 = 0.0

    print("\nStarting Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss, val_p, val_r, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        curr_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val P: {val_p * 100:.2f}% | "
            f"Val R: {val_r * 100:.2f}% | "
            f"Val F1: {val_f1 * 100:.2f}%"
        )

        with open(history_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_p, val_r, val_f1, curr_lr])

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "base_channels": 16,
                    "sigma": DEFAULT_SIGMA,
                    "val_f1": val_f1,
                },
                output_dir / "best_model.pt",
            )
            print(f"  --> Saved new best model (F1: {val_f1 * 100:.2f}%)")

    # Save final model
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "base_channels": 16,
            "sigma": DEFAULT_SIGMA,
            "val_f1": val_f1,
        },
        output_dir / "final_model.pt",
    )
    print(f"\nTraining Complete. Best Validation F1: {best_f1 * 100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Train JunctionUNet Keypoint Model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--loss", choices=["focal", "mse"], default="mse")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        loss_type=args.loss,
        max_samples=args.max_samples,
    )
