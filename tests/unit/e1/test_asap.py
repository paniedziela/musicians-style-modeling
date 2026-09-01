from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import mido

from musicians_style.e1.asap import (
    E1Config,
    build_e1_manifest,
    group_id_for,
)


def _write_midi(path: Path, pitch: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.Message("note_on", note=pitch, velocity=64, time=0))
    track.append(mido.Message("note_off", note=pitch, velocity=0, time=480))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(tmp_path: Path) -> E1Config:
    root = tmp_path / "asap"
    canonical = root / "Beethoven/Piano_Sonatas/17-1/midi_score.mid"
    alternate = root / "Beethoven/Piano_Sonatas/17-1_no_repeat/midi_score.mid"
    _write_midi(canonical)
    _write_midi(alternate, pitch=61)
    metadata = root / "metadata.csv"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    with metadata.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["composer", "title", "midi_score"])
        writer.writeheader()
        writer.writerow({"composer": "Beethoven", "title": "Piano_Sonatas_17-1", "midi_score": alternate.relative_to(root).as_posix()})
        writer.writerow({"composer": "Beethoven", "title": "Piano_Sonatas_17-1", "midi_score": canonical.relative_to(root).as_posix()})
    (root / "README.md").write_text("fixture", encoding="utf-8")
    (root / "LICENSE.md").write_text("fixture", encoding="utf-8")
    expected = {name: _sha(root / name) for name in ("metadata.csv", "README.md", "LICENSE.md")}
    return E1Config(
        dataset_root=root,
        output_dir=tmp_path / "derived",
        composers=("Beethoven",),
        minimum_samples_per_class=1,
        schema_version="test",
        expected_fingerprints=expected,
    )


def test_grouping_keeps_related_movements_together() -> None:
    assert group_id_for("Bach", "Prelude_bwv_846") == group_id_for("Bach", "Fugue_bwv_846")
    assert group_id_for("Beethoven", "Piano_Sonatas_17-1") == group_id_for("Beethoven", "Piano_Sonatas_17-3")
    assert group_id_for("Chopin", "Sonata_2_1st") == group_id_for("Chopin", "Sonata_2_4th")
    assert group_id_for("Chopin", "Ballades_1") != group_id_for("Chopin", "Ballades_2")


def test_manifest_prefers_score_without_no_repeat_suffix(tmp_path: Path) -> None:
    manifest, report = build_e1_manifest(_dataset(tmp_path))
    sample = manifest["samples"][0]
    assert sample["score_path"] == "Beethoven/Piano_Sonatas/17-1/midi_score.mid"
    assert sample["alternate_score_paths"] == ["Beethoven/Piano_Sonatas/17-1_no_repeat/midi_score.mid"]
    assert sample["validation_status"] == "accepted"
    assert sample["note_count"] == 1
    assert sample["length_beats"] == 1.0
    assert sample["length_bars"] == 0.25
    assert len(sample["legacy_features"]) == 42
    assert report["quality_gate"]["passed"] is True


def test_missing_score_is_documented_as_exclusion(tmp_path: Path) -> None:
    config = _dataset(tmp_path)
    (config.dataset_root / "Beethoven/Piano_Sonatas/17-1/midi_score.mid").unlink()
    manifest, report = build_e1_manifest(config)
    sample = manifest["samples"][0]
    assert sample["validation_status"] == "excluded"
    assert "does not exist" in sample["exclusion_reason"]
    assert report["excluded_count"] == 1
    assert report["quality_gate"]["passed"] is False
