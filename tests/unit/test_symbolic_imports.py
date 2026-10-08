import subprocess
import sys

import pytest

from musicians_style import evaluation
from musicians_style.evaluation.content import content_metrics
from musicians_style.midi.structure import Bar, build_bars
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent


def test_content_analysis_does_not_import_experiments_or_reporting():
    code = """
import sys
from musicians_style.content_metrics import measure_content, observe_midi
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.midi.types import InternalRepr, NoteEvent

piece = InternalRepr(480, (NoteEvent(0, 0, 60, 80, 480),))
observation = observe_midi(MidiPrettyPrinter().to_bytes(piece))
assert measure_content(observation, observation)['v2_exact_pitch']['status'] == 'passed'
for name in ('musicians_style.e1', 'musicians_style.e3', 'musicians_style.e4',
             'matplotlib', 'sklearn', 'scipy', 'torch'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_historical_structure_imports_are_the_same_objects():
    from musicians_style.e3.structure import build_bars as historical_build_bars
    from musicians_style.e3.types import Bar as HistoricalBar

    assert historical_build_bars is build_bars
    assert HistoricalBar is Bar


def test_reporting_exports_remain_available():
    for name in evaluation.__all__:
        assert getattr(evaluation, name) is not None
    with pytest.raises(AttributeError):
        evaluation.not_an_export


@pytest.mark.parametrize("resolution", [0, -1])
def test_content_metrics_reject_invalid_resolution(resolution):
    piece = InternalRepr(resolution, (NoteEvent(0, 0, 60, 80, 480),))
    with pytest.raises(ValueError, match="ticks_per_beat"):
        content_metrics(piece, piece)


def test_content_metrics_reject_invalid_meter_even_after_last_note():
    piece = InternalRepr(480, (NoteEvent(0, 0, 60, 80, 480),), (
        MetaEvent(960, "time_signature", {"numerator": 4, "denominator": 3}),
    ))
    with pytest.raises(ValueError, match="invalid time signature"):
        content_metrics(piece, piece)
