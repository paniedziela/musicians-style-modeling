"""E4.0--E4.1: audit and meter-aware onset/frame segmentation."""

from .segmentation import Segment, SegmentMap, encode_piece, stitch_segments
from .experiment import E4Config, audit_e4, load_e4_config

__all__ = ["E4Config", "Segment", "SegmentMap", "audit_e4", "encode_piece", "load_e4_config", "stitch_segments"]
