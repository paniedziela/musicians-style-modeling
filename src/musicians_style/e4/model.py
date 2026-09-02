"""Small conditional E4 generator and PatchGAN discriminator."""
from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.InstanceNorm2d(channels, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class ConditionalGenerator(nn.Module):
    """``G(x, c)`` mapping [B, 2, 64, 84] to two unbounded logits."""

    def __init__(self, *, num_classes: int = 3, conv_dim: int = 32, residual_blocks: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder = nn.Sequential(nn.Conv2d(2 + num_classes, conv_dim, 5, padding=2), nn.InstanceNorm2d(conv_dim, affine=True), nn.ReLU(inplace=True))
        self.residuals = nn.Sequential(*(ResidualBlock(conv_dim) for _ in range(residual_blocks)))
        self.decoder = nn.Conv2d(conv_dim, 2, 5, padding=2)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (2, 64, 84):
            raise ValueError("generator expects x with shape [B, 2, 64, 84]")
        if target.shape != (x.shape[0],):
            raise ValueError("target must have shape [B]")
        condition = torch.nn.functional.one_hot(target, self.num_classes).to(dtype=x.dtype, device=x.device)
        condition = condition[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return self.decoder(self.residuals(self.encoder(torch.cat((x, condition), dim=1))))


class PatchDiscriminator(nn.Module):
    """PatchGAN backbone with real/fake patches and a composer-class head."""

    def __init__(self, *, num_classes: int = 3, conv_dim: int = 32) -> None:
        super().__init__()
        channels = (conv_dim, conv_dim * 2, conv_dim * 4)
        layers: list[nn.Module] = []
        input_channels = 2
        for output_channels in channels:
            layers.extend((nn.Conv2d(input_channels, output_channels, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)))
            input_channels = output_channels
        self.backbone = nn.Sequential(*layers)
        self.real_fake = nn.Conv2d(input_channels, 1, 3, padding=1)
        self.composer = nn.Linear(input_channels, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1:] != (2, 64, 84):
            raise ValueError("discriminator expects x with shape [B, 2, 64, 84]")
        features = self.backbone(x)
        return self.real_fake(features), self.composer(features.mean(dim=(2, 3)))
