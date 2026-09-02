"""E4.0--E4.2: audit, segmentation, data isolation, and small conditional GAN."""

from .dataset import BalancedComposerSampler, PieceSegments, SegmentDataset, assert_split_isolation
from .segmentation import Segment, SegmentMap, encode_piece, stitch_segments
from .experiment import E4Config, audit_e4, load_e4_config
from .model import ConditionalGenerator, PatchDiscriminator
from .training import different_targets, gan_step

__all__ = ["BalancedComposerSampler", "ConditionalGenerator", "E4Config", "PatchDiscriminator", "PieceSegments", "Segment", "SegmentDataset", "SegmentMap", "assert_split_isolation", "audit_e4", "different_targets", "encode_piece", "gan_step", "load_e4_config", "stitch_segments"]
