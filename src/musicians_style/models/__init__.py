"""Model_GAN: StarGAN (warunkowany) i CycleGAN (per-artysta) (Wymagania 3.x)."""

from .stargan import ResidualBlock, StarGANDiscriminator, StarGANGenerator

__all__ = ["ResidualBlock", "StarGANGenerator", "StarGANDiscriminator"]
