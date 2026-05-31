"""Testy jednostkowe funkcji straty *Modelu_GAN* (zadanie 8.3, Wymaganie 3.12).

Weryfikują kontrakt funkcji z :mod:`musicians_style.models.losses` zgodnie z
sekcją *Funkcje straty* (``design.md``):

* wszystkie straty są **skalarami** (tensor 0-D),
* strata tożsamości i spójności cyklicznej wynosi **dokładnie 0**, gdy generator
  jest tożsamością i ``c_target == c_orig``,
* **gradienty przepływają** przez składniki strat (parametry generatora oraz
  logity dyskryminatora otrzymują niezerowy gradient),
* łączna strata stosuje wagi ``λ_cls = 1``, ``λ_cyc = 10``, ``λ_id = 5``.
"""

from __future__ import annotations

import torch

from musicians_style.models.losses import (
    DEFAULT_LAMBDA_CLS,
    DEFAULT_LAMBDA_CYC,
    DEFAULT_LAMBDA_ID,
    adversarial_loss,
    combined_generator_loss,
    cycle_consistency_loss,
    domain_classification_loss_fake,
    domain_classification_loss_real,
    generator_adversarial_loss,
    identity_loss,
)
from musicians_style.models.stargan import StarGANDiscriminator, StarGANGenerator

T, P = 64, 84


def _one_hot(batch: int, num_artists: int, indices: list[int]) -> torch.Tensor:
    c = torch.zeros(batch, num_artists)
    for row, idx in enumerate(indices):
        c[row, idx] = 1.0
    return c


class _IdentityGenerator:
    """Zaślepka generatora zwracająca wejście bez zmian: ``G(x, c) = x``."""

    def __call__(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return x


def _is_scalar(t: torch.Tensor) -> bool:
    return isinstance(t, torch.Tensor) and t.dim() == 0


# -- kształty (skalary) ------------------------------------------------------


def test_adversarial_loss_is_scalar() -> None:
    real = torch.rand(2, 1, 8, 8)
    fake = torch.rand(2, 1, 8, 8)
    assert _is_scalar(adversarial_loss(real, fake))
    assert _is_scalar(generator_adversarial_loss(fake))


def test_domain_classification_losses_are_scalars() -> None:
    logits = torch.randn(4, 3)
    c_onehot = _one_hot(4, 3, [0, 1, 2, 0])
    assert _is_scalar(domain_classification_loss_real(logits, c_onehot))
    assert _is_scalar(domain_classification_loss_fake(logits, c_onehot))


def test_domain_classification_accepts_index_labels() -> None:
    logits = torch.randn(3, 4)
    idx = torch.tensor([0, 2, 3])
    loss = domain_classification_loss_real(logits, idx)
    assert _is_scalar(loss)


def test_cycle_and_identity_losses_are_scalars() -> None:
    gen = StarGANGenerator(num_artists=3)
    x = torch.rand(2, 1, T, P)
    c_orig = _one_hot(2, 3, [0, 1])
    c_target = _one_hot(2, 3, [2, 0])
    assert _is_scalar(cycle_consistency_loss(gen, x, c_orig, c_target))
    assert _is_scalar(identity_loss(gen, x, c_orig))


# -- identity / cycle = 0 dla generatora tożsamościowego ---------------------


def test_identity_loss_zero_for_identity_generator() -> None:
    gen = _IdentityGenerator()
    x = torch.rand(2, 1, T, P)
    c_orig = _one_hot(2, 3, [0, 1])
    loss = identity_loss(gen, x, c_orig)
    assert torch.equal(loss, torch.zeros_like(loss))
    assert float(loss) == 0.0


def test_cycle_loss_zero_for_identity_generator() -> None:
    gen = _IdentityGenerator()
    x = torch.rand(2, 1, T, P)
    c_orig = _one_hot(2, 3, [0, 1])
    c_target = _one_hot(2, 3, [2, 0])
    loss = cycle_consistency_loss(gen, x, c_orig, c_target)
    assert float(loss) == 0.0


# -- przepływ gradientu ------------------------------------------------------


def test_adversarial_loss_gradients_flow_to_discriminator() -> None:
    disc = StarGANDiscriminator(num_artists=3)
    x_real = torch.rand(2, 1, T, P)
    x_fake = torch.rand(2, 1, T, P)
    src_real, _ = disc(x_real)
    src_fake, _ = disc(x_fake)
    loss = adversarial_loss(src_real, src_fake)
    loss.backward()
    grads = [p.grad for p in disc.parameters() if p.grad is not None]
    assert grads, "Brak gradientów w dyskryminatorze."
    assert any(torch.any(g != 0) for g in grads)


def test_classification_loss_gradients_flow() -> None:
    disc = StarGANDiscriminator(num_artists=4)
    x = torch.rand(3, 1, T, P)
    _, cls = disc(x)
    c = _one_hot(3, 4, [0, 1, 3])
    loss = domain_classification_loss_real(cls, c)
    loss.backward()
    grads = [p.grad for p in disc.parameters() if p.grad is not None]
    assert any(torch.any(g != 0) for g in grads)


def test_cycle_loss_gradients_flow_to_generator() -> None:
    gen = StarGANGenerator(num_artists=2)
    x = torch.rand(1, 1, T, P)
    c_orig = _one_hot(1, 2, [0])
    c_target = _one_hot(1, 2, [1])
    loss = cycle_consistency_loss(gen, x, c_orig, c_target)
    loss.backward()
    grads = [p.grad for p in gen.parameters() if p.grad is not None]
    assert grads, "Brak gradientów w generatorze."
    assert any(torch.any(g != 0) for g in grads)


# -- łączna strata generatora ------------------------------------------------


def test_combined_generator_loss_applies_default_weights() -> None:
    l_adv = torch.tensor(1.0)
    l_cls = torch.tensor(2.0)
    l_cyc = torch.tensor(3.0)
    l_id = torch.tensor(4.0)
    total = combined_generator_loss(l_adv, l_cls, l_cyc, l_id)
    expected = (
        1.0
        + DEFAULT_LAMBDA_CLS * 2.0
        + DEFAULT_LAMBDA_CYC * 3.0
        + DEFAULT_LAMBDA_ID * 4.0
    )
    assert float(total) == expected
    # Potwierdzenie wartości wag wymaganych przez Wymaganie 3.12.
    assert (DEFAULT_LAMBDA_CLS, DEFAULT_LAMBDA_CYC, DEFAULT_LAMBDA_ID) == (1.0, 10.0, 5.0)


def test_combined_generator_loss_custom_weights() -> None:
    total = combined_generator_loss(
        torch.tensor(1.0),
        torch.tensor(1.0),
        torch.tensor(1.0),
        torch.tensor(1.0),
        lambda_cls=2.0,
        lambda_cyc=3.0,
        lambda_id=4.0,
    )
    assert float(total) == 1.0 + 2.0 + 3.0 + 4.0


def test_combined_generator_loss_is_differentiable() -> None:
    gen = StarGANGenerator(num_artists=2)
    disc = StarGANDiscriminator(num_artists=2)
    x = torch.rand(1, 1, T, P)
    c_orig = _one_hot(1, 2, [0])
    c_target = _one_hot(1, 2, [1])

    fake = gen(x, c_target)
    src_fake, cls_fake = disc(fake)
    l_adv = generator_adversarial_loss(src_fake)
    l_cls = domain_classification_loss_fake(cls_fake, c_target)
    l_cyc = cycle_consistency_loss(gen, x, c_orig, c_target)
    l_id = identity_loss(gen, x, c_orig)

    total = combined_generator_loss(l_adv, l_cls, l_cyc, l_id)
    assert _is_scalar(total)
    total.backward()
    grads = [p.grad for p in gen.parameters() if p.grad is not None]
    assert any(torch.any(g != 0) for g in grads)
