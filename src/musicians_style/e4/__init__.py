"""E4: guarded conditional-GAN experiment from audit through outer test."""

from .dataset import BalancedComposerSampler, PieceSegments, SegmentDataset, assert_split_isolation
from .segmentation import Segment, SegmentMap, encode_piece, stitch_segments
from .experiment import E4Config, audit_e4, evaluate_e4, load_e4_config, prepare_e4, report_e4, smoke_e4, test_e4, train_e4
from .model import ConditionalGenerator, PatchDiscriminator
from .training import different_targets, gan_step

__all__ = ["BalancedComposerSampler", "ConditionalGenerator", "E4Config", "PatchDiscriminator", "PieceSegments", "Segment", "SegmentDataset", "SegmentMap", "assert_split_isolation", "audit_e4", "different_targets", "encode_piece", "evaluate_e4", "gan_step", "load_e4_config", "prepare_e4", "report_e4", "smoke_e4", "stitch_segments", "test_e4", "train_e4"]
