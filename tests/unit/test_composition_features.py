"""The reusable custom93 extractor must not load experiment runners."""

import subprocess
import sys

import numpy as np

from musicians_style.features.composition import FEATURE_SPECS, extract_composition_features
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent


def test_composition_extraction_without_experiment_or_classifier_imports():
    code = """
import sys
from musicians_style.features.composition import extract_composition_features
from musicians_style.midi.types import InternalRepr, NoteEvent

vector = extract_composition_features(InternalRepr(480, (NoteEvent(0, 0, 60, 80, 480),)))
assert vector.shape == (93,)
for name in ('musicians_style.e1', 'musicians_style.e2', 'musicians_style.e3',
             'musicians_style.e4', 'sklearn', 'matplotlib', 'torch'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_syncopation_uses_each_notes_duration_and_active_meter():
    piece = InternalRepr(480, (
        NoteEvent(240, 0, 60, 80, 120),  # Same onset, different syncopation.
        NoteEvent(240, 0, 64, 80, 360),
        NoteEvent(1920, 0, 62, 80, 480),
        NoteEvent(2040, 0, 65, 80, 240),  # Halfway through an eighth-note pulse.
    ), (
        MetaEvent(0, "time_signature", {"numerator": 4, "denominator": 4}),
        MetaEvent(1920, "time_signature", {"numerator": 6, "denominator": 8}),
    ))
    values = dict(zip((spec["name"] for spec in FEATURE_SPECS), extract_composition_features(piece)))
    assert values["syncopated_note_ratio"] == 0.5
    assert values["offbeat_onset_ratio"] == 2 / 3
    np.testing.assert_array_equal(
        extract_composition_features(piece),
        extract_composition_features(InternalRepr(480, tuple(reversed(piece.notes)), piece.meta)),
    )
