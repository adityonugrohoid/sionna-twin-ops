"""The surrogate network (spec M1, M1b): a small U-Net with two output heads."""

import torch
from torch import Tensor, nn

from sionna_twin_ops.features import INPUT_CHANNELS

RESIDUAL_SCALE_DB = 10.0  # the residual head predicts dB / 10


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3 x 3 convolutions, each with group norm and SiLU.

    Args:
        c_in: Input channels.
        c_out: Output channels.

    Returns:
        The block.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.SiLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.SiLU(),
    )


class UNet(nn.Module):
    """Four-level U-Net over a 128 x 128 map.

    Outputs two channels per cell: the path-gain residual over B0 (in units of
    RESIDUAL_SCALE_DB) and the logit that the ray tracer has power in the cell.
    """

    def __init__(self, width: int) -> None:
        """Build the network.

        Args:
            width: Channels at the first level; doubled at each level down.
        """
        super().__init__()
        widths = [width, 2 * width, 4 * width, 8 * width]
        self.down = nn.ModuleList()
        c_in = INPUT_CHANNELS
        for w in widths:
            self.down.append(conv_block(c_in, w))
            c_in = w
        self.bottom = conv_block(widths[-1], 2 * widths[-1])
        self.up = nn.ModuleList()
        self.merge = nn.ModuleList()
        c = 2 * widths[-1]
        for w in reversed(widths):
            self.up.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.merge.append(conv_block(2 * w, w))
            c = w
        self.head = nn.Conv2d(widths[0], 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Map inputs (batch, INPUT_CHANNELS, 128, 128) to outputs (batch, 2, 128, 128)."""
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = nn.functional.max_pool2d(x, 2)
        x = self.bottom(x)
        for up, merge, skip in zip(self.up, self.merge, reversed(skips), strict=True):
            x = merge(torch.cat([up(x), skip], dim=1))
        out: Tensor = self.head(x)
        return out


def parameter_count(model: nn.Module) -> int:
    """Trainable parameters of a model.

    Args:
        model: The model.

    Returns:
        The count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
