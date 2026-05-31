"""Testy jednostkowe modeli StarGAN (zadanie 8.1).

Weryfikują kontrakt :class:`~musicians_style.models.stargan.StarGANGenerator`
oraz :class:`~musicians_style.models.stargan.StarGANDiscriminator` z sekcji
*Model_GAN_Warunkowany* (``design.md``) i Wymagania 3.1, 3.12:

* generator ``G(x, c)`` przyjmuje pianoroll ``[B, 1, T, P]`` i *Etykietę_Artysty*
  one-hot ``[B, N]`` oraz zwraca pianoroll ``[B, 1, T, P]`` w zakresie ``[0, 1]``,
* dyskryminator PatchGAN zwraca krotkę ``(D_src, D_cls)`` z mapą real/fake oraz
  logitami klasyfikacji artysty ``[B, N]``.
"""

from __future__ import annotations

import torch

from musicians_style.models.stargan import (
    ResidualBlock,
    StarGANDiscriminator,
    StarGANGenerator,
)

T, P = 64, 84


def _one_hot(batch: int, num_artists: int, indices: list[int]) -> torch.Tensor:
    c = torch.zeros(batch, num_artists)
    for row, idx in enumerate(indices):
        c[row, idx] = 1.0
    return c


# -- StarGANGenerator --------------------------------------------------------


def test_generator_output_shape_matches_input() -> None:
    n = 3
    gen = StarGANGenerator(num_artists=n)
    x = torch.rand(2, 1, T, P)
    c = _one_hot(2, n, [0, 2])
    y = gen(x, c)
    assert y.shape == (2, 1, T, P)


def test_generator_output_in_unit_interval() -> None:
    gen = StarGANGenerator(num_artists=4)
    x = torch.rand(3, 1, T, P)
    c = _one_hot(3, 4, [0, 1, 3])
    y = gen(x, c)
    assert torch.all(y >= 0.0) and torch.all(y <= 1.0)


def test_generator_accepts_3d_input() -> None:
    # Wejście [B, T, P] jest automatycznie rozszerzane do [B, 1, T, P].
    gen = StarGANGenerator(num_artists=2)
    x = torch.rand(2, T, P)
    c = _one_hot(2, 2, [0, 1])
    y = gen(x, c)
    assert y.shape == (2, 1, T, P)


def test_generator_label_changes_output() -> None:
    # Różne Etykiety_Artysty dają różne wyjścia dla tego samego wejścia.
    torch.manual_seed(0)
    gen = StarGANGenerator(num_artists=3)
    x = torch.rand(1, 1, T, P)
    y0 = gen(x, _one_hot(1, 3, [0]))
    y1 = gen(x, _one_hot(1, 3, [1]))
    assert not torch.allclose(y0, y1)


def test_generator_respects_n_residual_blocks() -> None:
    gen = StarGANGenerator(num_artists=2, n_residual_blocks=3)
    n_blocks = sum(1 for m in gen.modules() if isinstance(m, ResidualBlock))
    assert n_blocks == 3


def test_residual_block_preserves_shape() -> None:
    block = ResidualBlock(dim=8)
    x = torch.rand(2, 8, 16, 16)
    assert block(x).shape == x.shape


# -- StarGANDiscriminator ----------------------------------------------------


def test_discriminator_returns_two_heads() -> None:
    n = 3
    disc = StarGANDiscriminator(num_artists=n)
    x = torch.rand(2, 1, T, P)
    out = disc(x)
    assert isinstance(out, tuple) and len(out) == 2


def test_discriminator_src_is_patch_map() -> None:
    disc = StarGANDiscriminator(num_artists=3)
    x = torch.rand(2, 1, T, P)
    src, _ = disc(x)
    # Mapa PatchGAN: jeden kanał, zredukowane wymiary przestrzenne.
    assert src.dim() == 4
    assert src.shape[0] == 2 and src.shape[1] == 1


def test_discriminator_cls_logits_shape() -> None:
    n = 5
    disc = StarGANDiscriminator(num_artists=n)
    x = torch.rand(4, 1, T, P)
    _, cls = disc(x)
    assert cls.shape == (4, n)


def test_discriminator_cls_shape_stable_for_other_input_size() -> None:
    # Głowica D_cls zwraca [B, N] niezależnie od wymiarów wejścia.
    n = 3
    disc = StarGANDiscriminator(num_artists=n, input_size=(T, P))
    _, cls = disc(torch.rand(2, 1, 32, 48))
    assert cls.shape == (2, n)


def test_discriminator_accepts_3d_input() -> None:
    disc = StarGANDiscriminator(num_artists=2)
    _, cls = disc(torch.rand(2, T, P))
    assert cls.shape == (2, 2)
