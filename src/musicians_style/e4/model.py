"""Small conditional E4 generator and PatchGAN discriminator."""
from __future__ import annotations

import torch
from torch import nn


class ConditionalInstanceNorm2d(nn.Module):
    """Instance normalization whose affine parameters come from the target domain.

    A spatially constant one-hot plane followed by ordinary InstanceNorm loses
    almost all conditioning information.  Supplying target-specific scale and
    bias *after* normalization keeps that information available at every cell.
    """

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.channels = channels
        self.normalization = nn.InstanceNorm2d(channels, affine=False)
        self.affine = nn.Embedding(num_classes, 2 * channels)
        nn.init.normal_(self.affine.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scale, bias = self.affine(target).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        bias = bias[:, :, None, None]
        return self.normalization(x) * (1 + scale) + bias


class ConditionalResidualBlock(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect")
        self.norm1 = ConditionalInstanceNorm2d(channels, num_classes)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect")
        self.norm2 = ConditionalInstanceNorm2d(channels, num_classes)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.norm1(self.conv1(x), target))
        return x + self.norm2(self.conv2(residual), target)


class ConditionalGenerator(nn.Module):
    """``G(x, c)`` mapping [B, 2, 64, 84] to two unbounded logits."""

    def __init__(self, *, num_classes: int = 3, conv_dim: int = 32, residual_blocks: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder = nn.Conv2d(2, conv_dim, 5, padding=2, padding_mode="reflect")
        self.encoder_norm = ConditionalInstanceNorm2d(conv_dim, num_classes)
        self.residuals = nn.ModuleList(
            ConditionalResidualBlock(conv_dim, num_classes) for _ in range(residual_blocks)
        )
        self.activation = nn.ReLU(inplace=True)
        self.decoder = nn.Conv2d(conv_dim, 2, 5, padding=2, padding_mode="reflect")
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (2, 64, 84):
            raise ValueError("generator expects x with shape [B, 2, 64, 84]")
        if target.shape != (x.shape[0],):
            raise ValueError("target must have shape [B]")
        features = self.activation(self.encoder_norm(self.encoder(x), target))
        for block in self.residuals:
            features = block(features, target)
        # Start from a conservative identity mapping instead of an untrained
        # dense pianoroll.  The learned residual can still add or remove notes.
        identity_logits = torch.logit(x.clamp(0.05, 0.95))
        return identity_logits + self.decoder(features)


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
