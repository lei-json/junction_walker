#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import IntersectionUNet


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_MODEL = Path(
    "training_output/best_model.pt"
)

DEFAULT_OUTPUT_DIR = Path(
    "prediction_output"
)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            f"Using CUDA: "
            f"{torch.cuda.get_device_name(0)}"
        )

        return device

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):

        print(
            "Using Apple MPS."
        )

        return torch.device("mps")

    print(
        "Using CPU."
    )

    return torch.device("cpu")


# ============================================================
# LOAD IMAGE
# ============================================================

def load_grayscale(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:

        raise RuntimeError(
            f"Could not load image: {path}"
        )

    return image


# ============================================================
# PAD IMAGE
# ============================================================

def pad_image(
    image,
    target_height,
    target_width,
):
    """
    Pad an image on the bottom and right.

    The original pixels are kept at exactly the same
    coordinates.

    No resizing is performed.
    """

    height, width = image.shape

    if (
        height > target_height
        or width > target_width
    ):

        raise ValueError(
            f"Input image is "
            f"{width}x{height}, but the model expects "
            f"at most "
            f"{target_width}x{target_height}.\n"
            f"Image must not be resized because that would "
            f"change the coordinate system."
        )

    padded = np.zeros(
        (
            target_height,
            target_width,
        ),
        dtype=np.uint8,
    )

    padded[
        :height,
        :width,
    ] = image

    return padded


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    checkpoint_path,
    device,
):
    """
    Load the trained U-Net and information about the
    training canvas.
    """

    print()
    print(
        f"Loading model: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    base_channels = checkpoint.get(
        "base_channels",
        32,
    )

    target_height = checkpoint.get(
        "target_height"
    )

    target_width = checkpoint.get(
        "target_width"
    )

    sigma = checkpoint.get(
        "sigma",
        3.0,
    )

    if (
        target_height is None
        or target_width is None
    ):

        raise RuntimeError(
            "Checkpoint does not contain "
            "'target_height' and 'target_width'."
        )

    model = IntersectionUNet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        f"Model epoch: "
        f"{checkpoint.get('epoch', '?')}"
    )

    print(
        f"Training canvas: "
        f"{target_width}x{target_height}"
    )

    print(
        f"Base channels: "
        f"{base_channels}"
    )

    print(
        f"Heatmap sigma: "
        f"{sigma}"
    )

    return (
        model,
        target_height,
        target_width,
    )


# ============================================================
# PREDICT
# ============================================================

@torch.no_grad()
def predict(
    model,
    image,
    device,
    target_height,
    target_width,
):
    """
    Run the model on an image.

    The image is padded on the bottom/right to the exact
    training canvas size.

    Returns the full prediction canvas.
    """

    padded = pad_image(
        image,
        target_height,
        target_width,
    )

    image_float = (
        padded.astype(
            np.float32
        )
        / 255.0
    )

    tensor = torch.from_numpy(
        image_float
    ).unsqueeze(
        0
    ).unsqueeze(
        0
    )

    tensor = tensor.to(
        device
    )

    prediction = model(
        tensor
    )

    prediction = torch.sigmoid(
        prediction
    )

    prediction = (
        prediction[0, 0]
        .cpu()
        .numpy()
    )

    return (
        padded,
        prediction,
    )


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(
    original,
    padded,
    prediction,
    output_path,
):
    """
    Save:

        1. Original image
        2. CNN prediction
        3. Prediction overlaid on image

    The original image is displayed without the padding.
    """

    height, width = original.shape

    prediction_cropped = prediction[
        :height,
        :width,
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    axes[0].imshow(
        original,
        cmap="gray",
    )

    axes[0].set_title(
        "Input"
    )

    axes[0].axis(
        "off"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    axes[1].imshow(
        prediction_cropped,
        cmap="hot",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "CNN prediction"
    )

    axes[1].axis(
        "off"
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    axes[2].imshow(
        original,
        cmap="gray",
    )

    axes[2].imshow(
        prediction_cropped,
        cmap="hot",
        alpha=0.5,
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        "Prediction overlay"
    )

    axes[2].axis(
        "off"
    )

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
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run a trained intersection U-Net "
            "on a wing image."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Input wing image.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Trained .pt checkpoint.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output visualization path. "
            "Defaults to "
            "prediction_output/<input_name>.png."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not args.image.is_file():

        raise FileNotFoundError(
            f"Input image does not exist: "
            f"{args.image}"
        )

    if not args.model.is_file():

        raise FileNotFoundError(
            f"Model does not exist: "
            f"{args.model}"
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    print()
    print(
        f"Loading image: {args.image}"
    )

    image = load_grayscale(
        args.image
    )

    height, width = image.shape

    print(
        f"Input image size: "
        f"{width}x{height}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        model,
        target_height,
        target_width,
    ) = load_model(
        args.model,
        device,
    )

    # --------------------------------------------------------
    # Check whether padding is required
    # --------------------------------------------------------

    print()
    print(
        f"Model input size: "
        f"{target_width}x{target_height}"
    )

    if (
        width == target_width
        and height == target_height
    ):

        print(
            "Image already has the correct size."
        )

    else:

        print(
            "Padding image "
            "(no resizing)..."
        )

        print(
            f"  width:  "
            f"{width} -> {target_width}"
        )

        print(
            f"  height: "
            f"{height} -> {target_height}"
        )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print()
    print(
        "Running prediction..."
    )

    padded, prediction = predict(
        model,
        image,
        device,
        target_height,
        target_width,
    )

    # --------------------------------------------------------
    # Prediction statistics
    # --------------------------------------------------------

    print()
    print(
        "Prediction:"
    )

    print(
        f"  min  = "
        f"{prediction.min():.6f}"
    )

    print(
        f"  max  = "
        f"{prediction.max():.6f}"
    )

    print(
        f"  mean = "
        f"{prediction.mean():.6f}"
    )

    # --------------------------------------------------------
    # Determine output path
    # --------------------------------------------------------

    if args.output is None:

        output_dir = DEFAULT_OUTPUT_DIR

        output_path = (
            output_dir
            / f"{args.image.stem}.png"
        )

    else:

        output_path = args.output

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_visualization(
        image,
        padded,
        prediction,
        output_path,
    )

    print()
    print(
        f"Saved prediction: "
        f"{output_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

