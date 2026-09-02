from __future__ import annotations

import torch

from musicians_style.e4.model import ConditionalGenerator, PatchDiscriminator
from musicians_style.e4.training import gan_step


def test_gan_step_has_valid_shapes_distinct_targets_finite_losses_and_gradients() -> None:
    torch.manual_seed(1729)
    generator = ConditionalGenerator()
    discriminator = PatchDiscriminator()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    batch = {"x": torch.rand(3, 2, 64, 84), "mask": torch.ones(3, 64), "source": torch.tensor([0, 1, 2], dtype=torch.long)}
    assert generator(batch["x"], batch["source"]).shape == (3, 2, 64, 84)
    patch, classes = discriminator(batch["x"])
    assert patch.shape[:2] == (3, 1) and classes.shape == (3, 3)
    metrics = gan_step(generator, discriminator, generator_optimizer, discriminator_optimizer, batch, target_generator=torch.Generator().manual_seed(7))
    assert torch.all(metrics.target != batch["source"])
    assert all(torch.isfinite(torch.tensor(value)) for value in (metrics.discriminator_loss, metrics.generator_loss, metrics.adversarial_loss, metrics.classification_loss, metrics.cycle_loss, metrics.identity_loss))
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in generator.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in discriminator.parameters())
