from __future__ import annotations

import pytest
from pathlib import Path

import torch

from musicians_style.e4.dataset import PieceSegments
from musicians_style.e4.experiment import E4Config, _atomic_checkpoint, _checkpoint, _infer_piece, _load_models, load_e4_config
from musicians_style.e4.model import ConditionalGenerator, PatchDiscriminator
from musicians_style.e4.segmentation import encode_piece
from musicians_style.midi.parser import MidiParser
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent

pytestmark = pytest.mark.regression


def test_checkpoint_to_full_midi_to_parse(tmp_path) -> None:
    config = E4Config(None, None, None, tmp_path)
    source = InternalRepr(480, (NoteEvent(0, 0, 60, 90, 480), NoteEvent(4 * 1920 + 120, 0, 67, 90, 700)), (MetaEvent(0, "time_signature", {"numerator": 4, "denominator": 4}),), 1)
    generator, discriminator = ConditionalGenerator(), PatchDiscriminator()
    go = torch.optim.Adam(generator.parameters(), lr=.0002)
    do = torch.optim.Adam(discriminator.parameters(), lr=.0002)
    checkpoint = tmp_path / "best.pt"
    _atomic_checkpoint(checkpoint, _checkpoint(config, 1, generator, discriminator, go, do, 0.0))
    model = _load_models(config, checkpoint, torch.device("cpu"))
    rendered, _ = _infer_piece(model, PieceSegments("synthetic", "g", "Bach", "validation", encode_piece(source)), 1, torch.device("cpu"))
    output = tmp_path / "output.mid"
    MidiPrettyPrinter().write(rendered, output)
    reparsed = MidiParser().parse(output)
    assert reparsed.meta == source.meta
    assert max(note.tick + note.duration_ticks for note in reparsed.notes) == encode_piece(source).end_tick


def test_v3_config_is_isolated_from_historical_run() -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_e4_config(root / "configs" / "e4_asap_v3.yaml")
    assert config.run_dir == root / "experiments" / "e4_asap_v3"
    assert config.summary_path == root / "docs" / "results" / "E4_v3.md"
    assert config.min_epochs == config.patience == 10
    assert config.maximum_positive_weight == 12
