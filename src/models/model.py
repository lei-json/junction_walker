#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DOUBLE CONVOLUTION
# ============================================================

class DoubleConv(nn.Module):
    """
    Two 3x3 convolutions with BatchNorm and ReLU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels,
            ),
            nn.ReLU(
                inplace=True,
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels,
            ),
            nn.ReLU(
                inplace=True,
            ),
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# DOWN BLOCK
# ============================================================

class Down(nn.Module):
    """
    Downsampling block.

    MaxPool2d reduces spatial resolution by 2,
    followed by two convolutions.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,
            ),
            DoubleConv(
                in_channels,
                out_channels,
            ),
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# UP BLOCK
# ============================================================

class Up(nn.Module):
    """
    Upsampling block with a U-Net skip connection.
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            (in_channels // 2) + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x,
        skip,
    ):

        x = self.up(x)

        # ----------------------------------------------------
        # Account for odd image dimensions
        # ----------------------------------------------------

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_x != 0 or diff_y != 0:

            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        # ----------------------------------------------------
        # Concatenate skip connection
        # ----------------------------------------------------

        x = torch.cat(
            [
                skip,
                x,
            ],
            dim=1,
        )

        return self.conv(x)


# ============================================================
# INTERSECTION U-NET
# ============================================================

class IntersectionUNet(nn.Module):
    """
    U-Net for full-wing intersection detection.

    Input:
        B x 1 x H x W

    Output:
        B x 1 x H x W

    The output contains logits for an intersection
    probability heatmap.

    Apply torch.sigmoid() to obtain probabilities.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.input_block = DoubleConv(
            in_channels,
            base_channels,
        )

        self.down1 = Down(
            base_channels,
            base_channels * 2,
        )

        self.down2 = Down(
            base_channels * 2,
            base_channels * 4,
        )

        self.down3 = Down(
            base_channels * 4,
            base_channels * 8,
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = Down(
            base_channels * 8,
            base_channels * 16,
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.up1 = Up(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
        )

        self.up2 = Up(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
        )

        self.up3 = Up(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
        )

        self.up4 = Up(
            base_channels * 2,
            base_channels,
            base_channels,
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        self.output = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        x1 = self.input_block(x)

        x2 = self.down1(
            x1
        )

        x3 = self.down2(
            x2
        )

        x4 = self.down3(
            x3
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        x5 = self.bottleneck(
            x4
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        x = self.up1(
            x5,
            x4,
        )

        x = self.up2(
            x,
            x3,
        )

        x = self.up3(
            x,
            x2,
        )

        x = self.up4(
            x,
            x1,
        )

        # ----------------------------------------------------
        # Final logits
        # ----------------------------------------------------

        x = self.output(
            x
        )

        return x


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Creating IntersectionUNet..."
    )

    model = IntersectionUNet()

    print(
        model
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        f"Trainable parameters: "
        f"{parameters:,}"
    )

    # --------------------------------------------------------
    # Test input
    # --------------------------------------------------------

    height = 512
    width = 512

    test = torch.randn(
        1,
        1,
        height,
        width,
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            test
        )

    # --------------------------------------------------------
    # Print shapes
    # --------------------------------------------------------

    print()
    print(
        f"Input shape:  "
        f"{tuple(test.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )

    # --------------------------------------------------------
    # Convert logits to probabilities
    # --------------------------------------------------------

    probability = torch.sigmoid(
        output
    )

    print()
    print(
        f"Probability range: "
        f"{probability.min().item():.4f} "
        f"-> "
        f"{probability.max().item():.4f}"
    )

    print()
    print(
        "Model test successful."
    )
