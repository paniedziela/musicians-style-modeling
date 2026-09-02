"""One verified E4 GAN update; harness/checkpoint policy belongs to E4.3."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import ConditionalGenerator, PatchDiscriminator


def different_targets(source: torch.Tensor, *, num_classes: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample a target label uniformly from all labels except the source label."""
    if source.ndim != 1 or source.dtype != torch.long:
        raise ValueError("source must be a rank-1 LongTensor")
    if num_classes < 2 or bool(((source < 0) | (source >= num_classes)).any()):
        raise ValueError("invalid composer labels")
    shifts = torch.randint(1, num_classes, source.shape, device=source.device, generator=generator)
    target = (source + shifts) % num_classes
    if bool((target == source).any()):
        raise RuntimeError("target composer must differ from source")
    return target


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4 or mask.shape != prediction.shape[:1] + prediction.shape[2:3]:
        raise ValueError("invalid prediction, target, or [B, 64] mask shape")
    weights = mask[:, None, :, None].to(dtype=prediction.dtype)
    return ((prediction - target).abs() * weights).sum() / (weights.sum() * prediction.shape[1] * prediction.shape[3]).clamp_min(1)


@dataclass(frozen=True)
class StepMetrics:
    discriminator_loss: float
    generator_loss: float
    adversarial_loss: float
    classification_loss: float
    cycle_loss: float
    identity_loss: float
    target: torch.Tensor


def gan_step(
    generator: ConditionalGenerator, discriminator: PatchDiscriminator, generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer, batch: dict[str, torch.Tensor], *, num_classes: int = 3,
    lambda_cls: float = 1.0, lambda_cyc: float = 10.0, lambda_id: float = 10.0,
    target_generator: torch.Generator | None = None,
) -> StepMetrics:
    x, mask, source = batch["x"], batch["mask"], batch["source"]
    target = different_targets(source, num_classes=num_classes, generator=target_generator)

    discriminator_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake = torch.sigmoid(generator(x, target))
    real_patch, real_class = discriminator(x)
    fake_patch, _ = discriminator(fake)
    discriminator_loss = ((real_patch - 1).square().mean() + fake_patch.square().mean() + F.cross_entropy(real_class, source))
    discriminator_loss.backward()
    discriminator_optimizer.step()

    generator_optimizer.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    fake_logits = generator(x, target)
    fake = torch.sigmoid(fake_logits)
    fake_patch, fake_class = discriminator(fake)
    adversarial_loss = (fake_patch - 1).square().mean()
    classification_loss = F.cross_entropy(fake_class, target)
    cycled = torch.sigmoid(generator(fake, source))
    identity = torch.sigmoid(generator(x, source))
    cycle_loss = masked_l1(cycled, x, mask)
    identity_loss = masked_l1(identity, x, mask)
    generator_loss = adversarial_loss + lambda_cls * classification_loss + lambda_cyc * cycle_loss + lambda_id * identity_loss
    generator_loss.backward()
    generator_optimizer.step()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    values = (discriminator_loss, generator_loss, adversarial_loss, classification_loss, cycle_loss, identity_loss)
    if not all(bool(torch.isfinite(value)) for value in values):
        raise FloatingPointError("non-finite E4 training loss")
    return StepMetrics(*(float(value.detach()) for value in values), target.detach())
