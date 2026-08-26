#!/usr/bin/env python3

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import IntersectionDataset
from model import IntersectionUNet


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_HEATMAP_SIGMA = 3.0
DEFAULT_BASE_CHANNELS = 32
DEFAULT_SEED = 42

DEFAULT_OUTPUT_DIR = Path(
    "training_output"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():
    """
    Select the best available device.
    """

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        print(
            f"Using CUDA: "
            f"{torch.cuda.get_device_name(0)}"
        )

        return device

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):

        device = torch.device(
            "mps"
        )

        print(
            "Using Apple MPS."
        )

        return device

    print(
        "Using CPU."
    )

    return torch.device(
        "cpu"
    )


# ============================================================
# WEIGHTED BCE LOSS
# ============================================================

class WeightedBCELoss(nn.Module):
    """
    Weighted binary cross entropy.

    Intersection heatmaps are sparse, so background pixels
    greatly outnumber intersection pixels.
    """

    def __init__(
        self,
        positive_weight=10.0,
    ):

        super().__init__()

        self.positive_weight = (
            positive_weight
        )

    def forward(
        self,
        prediction,
        target,
    ):

        weight = torch.ones_like(
            target
        )

        weight = torch.where(
            target > 0.1,
            self.positive_weight,
            weight,
        )

        return nn.functional.binary_cross_entropy_with_logits(
            prediction,
            target,
            weight=weight,
        )


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):
    """
    Dice loss for sparse spatial predictions.
    """

    def __init__(
        self,
        smooth=1.0,
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        prediction,
        target,
    ):

        prediction = torch.sigmoid(
            prediction
        )

        prediction = prediction.reshape(
            prediction.shape[0],
            -1,
        )

        target = target.reshape(
            target.shape[0],
            -1,
        )

        intersection = (
            prediction * target
        ).sum(
            dim=1
        )

        denominator = (
            prediction.sum(
                dim=1
            )
            + target.sum(
                dim=1
            )
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        return (
            1.0 - dice
        ).mean()


# ============================================================
# COMBINED LOSS
# ============================================================

class IntersectionLoss(nn.Module):
    """
    BCE + Dice loss.
    """

    def __init__(
        self,
        positive_weight=10.0,
        dice_weight=1.0,
    ):

        super().__init__()

        self.bce = WeightedBCELoss(
            positive_weight=positive_weight,
        )

        self.dice = DiceLoss()

        self.dice_weight = (
            dice_weight
        )

    def forward(
        self,
        prediction,
        target,
    ):

        bce = self.bce(
            prediction,
            target,
        )

        dice = self.dice(
            prediction,
            target,
        )

        return (
            bce
            + self.dice_weight * dice
        )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        predictions = model(
            images
        )

        loss = criterion(
            predictions,
            targets,
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

    return (
        total_loss
        / len(loader.dataset)
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        predictions = model(
            images
        )

        loss = criterion(
            predictions,
            targets,
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

    return (
        total_loss
        / len(loader.dataset)
    )


# ============================================================
# SAVE VALIDATION EXAMPLE
# ============================================================

@torch.no_grad()
def save_validation_example(
    model,
    dataset,
    index,
    device,
    output_path,
):

    model.eval()

    image, target = dataset[
        index
    ]

    input_tensor = (
        image
        .unsqueeze(0)
        .to(device)
    )

    prediction = model(
        input_tensor
    )

    prediction = torch.sigmoid(
        prediction
    )

    prediction = (
        prediction[0, 0]
        .cpu()
        .numpy()
    )

    image = (
        image[0]
        .cpu()
        .numpy()
    )

    target = (
        target[0]
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    axes[0].imshow(
        image,
        cmap="gray",
    )

    axes[0].set_title(
        "Input wing"
    )

    axes[1].imshow(
        target,
        cmap="hot",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "Target intersections"
    )

    axes[2].imshow(
        prediction,
        cmap="hot",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        "CNN prediction"
    )

    for axis in axes:

        axis.axis("off")

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# SAVE TRAINING CURVE
# ============================================================

def save_training_curve(
    history,
    output_path,
):

    epochs = [
        row["epoch"]
        for row in history
    ]

    train_losses = [
        row["train_loss"]
        for row in history
    ]

    validation_losses = [
        row["validation_loss"]
        for row in history
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        epochs,
        train_losses,
        label="Training loss",
    )

    ax.plot(
        epochs,
        validation_losses,
        label="Validation loss",
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.set_title(
        "Training History"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(
        fig
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train a U-Net to predict "
            "wing intersection heatmaps."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    parser.add_argument(
        "--saved-wings",
        type=Path,
        default=Path(
            "../../data/saved_wings"
        ),
        help=(
            "Directory containing the "
            "selected training wings."
        ),
    )

    parser.add_argument(
        "--seeds",
        type=Path,
        default=Path(
            "../../data/seeds"
        ),
        help=(
            "Directory containing "
            "intersection .npy files."
        ),
    )

    parser.add_argument(
        "--raw",
        type=Path,
        default=Path(
            "../../data/raw"
        ),
        help=(
            "Directory containing the "
            "original raw wing images."
        ),
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_HEATMAP_SIGMA,
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=DEFAULT_BASE_CHANNELS,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if args.epochs <= 0:

        raise ValueError(
            "--epochs must be greater than 0."
        )

    if args.batch_size <= 0:

        raise ValueError(
            "--batch-size must be greater than 0."
        )

    if args.learning_rate <= 0:

        raise ValueError(
            "--learning-rate must be greater than 0."
        )

    if not (
        0.0
        < args.validation_split
        < 1.0
    ):

        raise ValueError(
            "--validation-split must be "
            "between 0 and 1."
        )

    if args.sigma <= 0:

        raise ValueError(
            "--sigma must be greater than 0."
        )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # DATASET
    # ========================================================

    print()
    print(
        "Loading dataset..."
    )

    dataset = IntersectionDataset(
        saved_wings_dir=args.saved_wings,
        seeds_dir=args.seeds,
        raw_dir=args.raw,
        sigma=args.sigma,
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    validation_size = max(
        1,
        int(
            len(dataset)
            * args.validation_split
        ),
    )

    train_size = (
        len(dataset)
        - validation_size
    )

    if train_size <= 0:

        raise RuntimeError(
            "Not enough samples for "
            "training and validation."
        )

    train_dataset, validation_dataset = (
        random_split(
            dataset,
            [
                train_size,
                validation_size,
            ],
            generator=(
                torch.Generator()
                .manual_seed(
                    args.seed
                )
            ),
        )
    )

    print()

    print(
        f"Training samples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset):,}"
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Creating model..."
    )

    model = IntersectionUNet(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = IntersectionLoss(
        positive_weight=10.0,
        dice_weight=1.0,
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
    )

    # ========================================================
    # TRAINING
    # ========================================================

    best_validation_loss = float(
        "inf"
    )

    history = []

    print()
    print(
        "Starting training..."
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        validation_loss = validate(
            model,
            validation_loader,
            criterion,
            device,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": (
                    validation_loss
                ),
                "learning_rate": current_lr,
            }
        )

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{args.epochs:03d} | "
            f"train="
            f"{train_loss:.6f} | "
            f"val="
            f"{validation_loss:.6f} | "
            f"lr="
            f"{current_lr:.2e}"
        )

        # ====================================================
        # BEST MODEL
        # ====================================================

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            checkpoint = {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "validation_loss":
                    validation_loss,

                "train_loss":
                    train_loss,

                "target_height":
                    dataset.target_height,

                "target_width":
                    dataset.target_width,

                "sigma":
                    args.sigma,

                "base_channels":
                    args.base_channels,
            }

            torch.save(
                checkpoint,
                args.output_dir
                / "best_model.pt",
            )

            print(
                "  Saved new best model."
            )

    # ========================================================
    # FINAL MODEL
    # ========================================================

    final_checkpoint = {
        "epoch": args.epochs,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "validation_loss":
            validation_loss,

        "train_loss":
            train_loss,

        "target_height":
            dataset.target_height,

        "target_width":
            dataset.target_width,

        "sigma":
            args.sigma,

        "base_channels":
            args.base_channels,
    }

    torch.save(
        final_checkpoint,
        args.output_dir
        / "final_model.pt",
    )

    # ========================================================
    # TRAINING HISTORY CSV
    # ========================================================

    history_path = (
        args.output_dir
        / "training_history.csv"
    )

    with open(
        history_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "validation_loss",
                "learning_rate",
            ],
        )

        writer.writeheader()

        writer.writerows(
            history
        )

    # ========================================================
    # TRAINING CURVE
    # ========================================================

    save_training_curve(
        history,
        args.output_dir
        / "training_curve.png",
    )

    # ========================================================
    # VALIDATION EXAMPLES
    # ========================================================

    print()
    print(
        "Saving validation examples..."
    )

    number_of_examples = min(
        5,
        len(validation_dataset),
    )

    for i in range(
        number_of_examples
    ):

        save_validation_example(
            model,
            validation_dataset,
            i,
            device,
            args.output_dir
            / f"validation_{i:02d}.png",
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print(
        "Training complete."
    )

    print()
    print(
        f"Best model:"
    )

    print(
        f"  {args.output_dir / 'best_model.pt'}"
    )

    print()
    print(
        f"Final model:"
    )

    print(
        f"  {args.output_dir / 'final_model.pt'}"
    )

    print()
    print(
        f"Training history:"
    )

    print(
        f"  {history_path}"
    )

    print()
    print(
        f"Training curve:"
    )

    print(
        f"  {args.output_dir / 'training_curve.png'}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
