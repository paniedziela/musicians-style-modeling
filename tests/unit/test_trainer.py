"""Testy jednostkowe :mod:`musicians_style.training.trainer` (zadanie 9.2).

Weryfikują kontrakt :class:`~musicians_style.training.trainer.GANTrainer` z
sekcji *Pipeline_Treningu* (``design.md``) oraz Wymagania 3.2, 3.4, 3.6, 3.8,
3.10, 3.11, 3.14:

* trening jednej epoki w trybie ``conditional`` (StarGAN) i ``per_artist``
  (CycleGAN) na drobnych plikach MIDI produkuje *Punkt_Kontrolny* zawierający
  wszystkie wymagane pola oraz poprawne metadane (``mode``, ``artists``,
  ``config_hash``, ``git_commit``, ``model_version``),
* zapis logów ``train.jsonl`` (straty per-iteracja, metryki walidacyjne),
* selektywne ostrzeżenia o niewystarczającej liczności plików w trybie
  warunkowanym (Wymaganie 3.11).

Testy używają bardzo małego modelu (``conv_dim=8``, jeden blok residualny,
dwie warstwy dyskryminatora) i małego pianorolla (``[1, 16, 16]``), trenowanego
przez jedną epokę na CPU - dzięki temu pozostają szybkie.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from mido import MidiFile, MidiTrack, Message, MetaMessage

from musicians_style.config import (
    Config,
    DatasetConfig,
    GeneratorConfig,
    ModelConfig,
    TrainingConfig,
)
from musicians_style.data.manifest import FileEntry, Manifest
from musicians_style.midi.pianoroll import Pianoroll
from musicians_style.training.dataset import MultiArtistManifest
from musicians_style.training.trainer import MODEL_VERSION, GANTrainer

_TICKS_PER_BEAT = 480

# Geometria drobnego pianorolla testowego: małe T i P dla szybkości.
_TINY_PITCH_RANGE = (48, 64)  # 16 wysokości
_TINY_WINDOW_STEPS = 16


def _write_midi(path: Path, pitch: int = 60, *, duration_ticks: int = 480) -> None:
    """Zapisuje minimalny poprawny plik SMF z pojedynczą nutą o danej wysokości."""
    mf = MidiFile(type=1, ticks_per_beat=_TICKS_PER_BEAT)
    conductor = MidiTrack()
    conductor.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    mf.tracks.append(conductor)
    notes = MidiTrack()
    notes.append(Message("note_on", note=pitch, velocity=100, time=0))
    notes.append(Message("note_off", note=pitch, velocity=0, time=duration_ticks))
    mf.tracks.append(notes)
    mf.save(str(path))


def _make_artist_dir(
    base: Path, artist_id: str, pitches: list[int]
) -> tuple[Path, Manifest]:
    """Tworzy katalog artysty z plikami MIDI i odpowiadający mu Manifest."""
    directory = base / artist_id
    directory.mkdir(parents=True, exist_ok=True)
    entries: list[FileEntry] = []
    for i, pitch in enumerate(pitches):
        name = f"track_{i:02d}.mid"
        _write_midi(directory / name, pitch=pitch)
        entries.append(
            FileEntry(
                path=name,
                sha256="0" * 64,
                duration_s=1.0,
                tracks=2,
                source="local",
            )
        )
    return directory, Manifest(artist_id=artist_id, files=entries)


def _tiny_config(
    mode: str,
    *,
    artists: list[str],
    reference_artist: str | None = None,
    min_files_per_artist: int = 30,
) -> Config:
    """Buduje minimalną :class:`Config` do szybkiego treningu CPU (1 epoka)."""
    return Config(
        seed=0,
        experiment_name="trainer_unit_test",
        dataset=DatasetConfig(min_files_per_artist=min_files_per_artist),
        model=ModelConfig(
            mode=mode,
            artists=artists,
            reference_artist=reference_artist,
            generator=GeneratorConfig(n_residual_blocks=1),
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.0002,
            optimizer="adam",
            beta1=0.5,
            beta2=0.999,
            device="cpu",
        ),
    )


def _tiny_pianoroll() -> Pianoroll:
    return Pianoroll(
        pitch_range=_TINY_PITCH_RANGE,
        steps_per_beat=4,
        window_steps=_TINY_WINDOW_STEPS,
    )


def _make_trainer(config: Config, mode: str, output_dir: Path, roots) -> GANTrainer:
    """Buduje trener z małym modelem (szybki na CPU)."""
    return GANTrainer(
        config,
        mode,  # type: ignore[arg-type]
        output_dir=output_dir,
        roots=roots,
        pianoroll=_tiny_pianoroll(),
        conv_dim=8,
        n_residual_blocks=1,
        disc_layers=2,
    )


_REQUIRED_CHECKPOINT_KEYS = {
    "epoch",
    "generator_state",
    "discriminator_state",
    "optimizer_g_state",
    "optimizer_d_state",
    "rng_states",
    "metadata",
    "history",
}


# -- tryb warunkowany (conditional / StarGAN) --------------------------------


def test_train_conditional_produces_checkpoint(tmp_path: Path) -> None:
    dir_a, m_a = _make_artist_dir(tmp_path / "data", "alpha", [50, 55])
    dir_b, m_b = _make_artist_dir(tmp_path / "data", "bravo", [52, 57])
    multi = MultiArtistManifest([m_a, m_b])

    output_dir = tmp_path / "exp"
    config = _tiny_config(
        "conditional", artists=["alpha", "bravo"], min_files_per_artist=1
    )
    trainer = _make_trainer(
        config, "conditional", output_dir, {"alpha": dir_a, "bravo": dir_b}
    )

    checkpoint_path = trainer.train(multi, seed=123)

    assert checkpoint_path.exists()
    assert checkpoint_path.name == "epoch_001.pt"
    assert checkpoint_path.parent == output_dir / "checkpoints"


def test_conditional_checkpoint_has_required_fields(tmp_path: Path) -> None:
    dir_a, m_a = _make_artist_dir(tmp_path / "data", "alpha", [50, 55])
    dir_b, m_b = _make_artist_dir(tmp_path / "data", "bravo", [52, 57])
    multi = MultiArtistManifest([m_a, m_b])

    config = _tiny_config(
        "conditional", artists=["alpha", "bravo"], min_files_per_artist=1
    )
    trainer = _make_trainer(
        config, "conditional", tmp_path / "exp", {"alpha": dir_a, "bravo": dir_b}
    )
    checkpoint_path = trainer.train(multi, seed=7)

    ckpt = torch.load(checkpoint_path)
    assert set(ckpt.keys()) >= _REQUIRED_CHECKPOINT_KEYS
    assert ckpt["epoch"] == 1

    # rng_states - cztery źródła losowości (Wymaganie 3.6).
    assert set(ckpt["rng_states"].keys()) == {
        "torch_cpu",
        "torch_cuda",
        "numpy",
        "python",
    }

    # history - serie strat treningu (Wymaganie 3.8).
    assert "loss_g" in ckpt["history"]
    assert "loss_d" in ckpt["history"]
    assert len(ckpt["history"]["loss_g"]) == 1


def test_conditional_checkpoint_metadata(tmp_path: Path) -> None:
    # Kolejność artystów w metadanych musi odpowiadać indeksom Etykiety_Artysty
    # (alfabetycznie), niezależnie od kolejności wejściowej.
    dir_q, m_q = _make_artist_dir(tmp_path / "data", "queen", [50])
    dir_a, m_a = _make_artist_dir(tmp_path / "data", "abba", [52])
    multi = MultiArtistManifest([m_q, m_a])

    config = _tiny_config(
        "conditional", artists=["abba", "queen"], min_files_per_artist=1
    )
    trainer = _make_trainer(
        config, "conditional", tmp_path / "exp", {"queen": dir_q, "abba": dir_a}
    )
    checkpoint_path = trainer.train(multi, seed=1)

    meta = torch.load(checkpoint_path)["metadata"]
    assert meta["mode"] == "conditional"
    assert meta["artists"] == ["abba", "queen"]
    assert meta["model_version"] == MODEL_VERSION
    assert meta["config_hash"].startswith("sha256:")
    assert "git_commit" in meta
    assert "created_at" in meta


def test_conditional_writes_train_jsonl(tmp_path: Path) -> None:
    import json

    dir_a, m_a = _make_artist_dir(tmp_path / "data", "alpha", [50, 55])
    dir_b, m_b = _make_artist_dir(tmp_path / "data", "bravo", [52, 57])
    multi = MultiArtistManifest([m_a, m_b])

    output_dir = tmp_path / "exp"
    config = _tiny_config(
        "conditional", artists=["alpha", "bravo"], min_files_per_artist=1
    )
    trainer = _make_trainer(
        config, "conditional", output_dir, {"alpha": dir_a, "bravo": dir_b}
    )
    trainer.train(multi, seed=5)

    log_path = output_dir / "logs" / "train.jsonl"
    assert log_path.exists()
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    # Każdy wpis ma wymagane pola formatu logu.
    for record in records:
        assert {"ts", "level", "component", "msg"} <= set(record.keys())
        assert record["component"] == "trainer"
    # Co najmniej jeden wpis to log iteracji ze stratami (Wymaganie 3.8).
    iter_records = [r for r in records if r["msg"] == "iteracja treningu"]
    assert iter_records
    assert "loss_g" in iter_records[0] and "loss_d" in iter_records[0]
    # Metryka walidacyjna po epoce.
    assert any(r["msg"] == "metryki walidacyjne epoki" for r in records)


# -- tryb per-artysta (per_artist / CycleGAN) --------------------------------


def test_train_per_artist_produces_checkpoint(tmp_path: Path) -> None:
    dir_solo, m_solo = _make_artist_dir(tmp_path / "data", "solo", [50, 55, 60])

    config = _tiny_config(
        "per_artist",
        artists=["solo"],
        reference_artist="control",
        min_files_per_artist=1,
    )
    trainer = _make_trainer(config, "per_artist", tmp_path / "exp", dir_solo)
    checkpoint_path = trainer.train(m_solo, seed=42)

    assert checkpoint_path.exists()
    ckpt = torch.load(checkpoint_path)
    assert set(ckpt.keys()) >= _REQUIRED_CHECKPOINT_KEYS
    meta = ckpt["metadata"]
    assert meta["mode"] == "per_artist"
    assert meta["artists"] == ["solo"]
    assert meta["model_version"] == MODEL_VERSION


# -- selektywne ostrzeżenia o liczności (Wymaganie 3.11) ---------------------


def test_selective_warning_only_for_artists_below_threshold(tmp_path: Path) -> None:
    # Próg domyślny 30: artysta 'low' ma 2 pliki (<30), 'high' ma 3 (też <30).
    _, m_low = _make_artist_dir(tmp_path / "data", "low", [50, 55])
    _, m_high = _make_artist_dir(tmp_path / "data", "high", [52, 57, 60])
    multi = MultiArtistManifest([m_low, m_high])

    config = _tiny_config(
        "conditional", artists=["high", "low"], min_files_per_artist=3
    )
    trainer = _make_trainer(config, "conditional", tmp_path / "exp", tmp_path / "data")

    insufficient = trainer.check_artist_counts(multi)
    # Próg 3: 'low' (2 pliki) jest poniżej, 'high' (3 pliki) nie.
    assert insufficient == ["low"]


def test_no_warning_when_all_sufficient(tmp_path: Path) -> None:
    _, m_a = _make_artist_dir(tmp_path / "data", "alpha", [50, 55])
    _, m_b = _make_artist_dir(tmp_path / "data", "bravo", [52, 57])
    multi = MultiArtistManifest([m_a, m_b])

    config = _tiny_config(
        "conditional", artists=["alpha", "bravo"], min_files_per_artist=2
    )
    trainer = _make_trainer(config, "conditional", tmp_path / "exp", tmp_path / "data")

    assert trainer.check_artist_counts(multi) == []


def test_train_records_insufficient_artists(tmp_path: Path) -> None:
    # Trening w trybie warunkowanym zapamiętuje listę artystów z ostrzeżeniem,
    # ale kontynuuje działanie i zapisuje Punkt_Kontrolny.
    dir_a, m_a = _make_artist_dir(tmp_path / "data", "alpha", [50, 55])
    dir_b, m_b = _make_artist_dir(tmp_path / "data", "bravo", [52, 57])
    multi = MultiArtistManifest([m_a, m_b])

    config = _tiny_config(
        "conditional", artists=["alpha", "bravo"], min_files_per_artist=30
    )
    trainer = _make_trainer(
        config, "conditional", tmp_path / "exp", {"alpha": dir_a, "bravo": dir_b}
    )
    checkpoint_path = trainer.train(multi, seed=0)

    assert checkpoint_path.exists()
    # Obaj artyści mają <30 plików → obaj na liście ostrzeżeń.
    assert trainer.insufficient_artists == ["alpha", "bravo"]


# -- walidacja parametrów ----------------------------------------------------


def test_invalid_mode_raises(tmp_path: Path) -> None:
    config = _tiny_config("conditional", artists=["alpha"], min_files_per_artist=1)
    with pytest.raises(ValueError):
        GANTrainer(config, "bogus", output_dir=tmp_path / "exp")  # type: ignore[arg-type]
