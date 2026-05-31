"""Model_GAN: StarGAN (warunkowany) i CycleGAN (per-artysta) (Wymagania 3.x)."""

from .cyclegan import CycleGANDiscriminator, CycleGANGenerator
from .losses import (
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
from .stargan import ResidualBlock, StarGANDiscriminator, StarGANGenerator

__all__ = [
    "ResidualBlock",
    "StarGANGenerator",
    "StarGANDiscriminator",
    "CycleGANGenerator",
    "CycleGANDiscriminator",
    "adversarial_loss",
    "generator_adversarial_loss",
    "domain_classification_loss_real",
    "domain_classification_loss_fake",
    "cycle_consistency_loss",
    "identity_loss",
    "combined_generator_loss",
    "DEFAULT_LAMBDA_CLS",
    "DEFAULT_LAMBDA_CYC",
    "DEFAULT_LAMBDA_ID",
]
