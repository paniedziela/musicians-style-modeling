from __future__ import annotations

import torch

from musicians_style.e4.model import ConditionalGenerator, PatchDiscriminator
from musicians_style.e4.training import estimate_positive_weights, gan_step, masked_l1


def test_conditioning_changes_interior_cells_and_identity_prior_is_sparse() -> None:
    torch.manual_seed(1729)
    generator = ConditionalGenerator()
    x = torch.zeros(1, 2, 64, 84)
    x[:, :, 20:24, 30:34] = 1
    bach = generator(x, torch.tensor([0]))
    chopin = generator(x, torch.tensor([2]))

    assert not torch.allclose(bach[:, :, 4:-4, 4:-4], chopin[:, :, 4:-4, 4:-4])
    decoded = torch.sigmoid(bach) >= .5
    assert torch.mean((decoded == x.bool()).float()) > .99


def test_weighted_reconstruction_penalizes_missed_onset_more_than_false_positive() -> None:
    mask = torch.ones(1, 1)
    weights = torch.tensor([10.0, 1.0])
    target_positive = torch.zeros(1, 2, 1, 2)
    target_positive[0, 0, 0, 0] = 1
    missed_positive = target_positive.clone()
    missed_positive[0, 0, 0, 0] = 0
    target_empty = torch.zeros_like(target_positive)
    false_positive = target_empty.clone()
    false_positive[0, 0, 0, 0] = 1

    positive_loss = masked_l1(missed_positive, target_positive, mask, positive_weights=weights)
    empty_loss = masked_l1(false_positive, target_empty, mask, positive_weights=weights)
    assert positive_loss > empty_loss


def test_positive_weights_are_train_derived_and_capped() -> None:
    class TinyDataset:
        rows = [
            {"x": torch.tensor([[[1, 0, 0]], [[1, 1, 0]]]), "mask": torch.ones(1)},
            {"x": torch.tensor([[[0, 0, 0]], [[0, 0, 1]]]), "mask": torch.ones(1)},
        ]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    assert torch.equal(estimate_positive_weights(TinyDataset(), maximum=4), torch.tensor([4.0, 1.0]))


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
