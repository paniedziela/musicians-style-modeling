"""Testy jednostkowe :mod:`musicians_style.inference.pipeline` (zadania 10.1-10.3).

Weryfikują kontrakt :class:`~musicians_style.inference.pipeline.StyleTransferPipeline`
z sekcji *Pipeline inferencji* (``design.md``) oraz Wymagania 5.1, 5.2, 5.6-5.11:

* :meth:`StyleTransferPipeline.infer_gan` (10.1) - przepływ
  ``parse → from_internal → G(x, c) → to_internal(template) → write`` z tworzeniem
  poprawnego pliku MIDI oraz zachowaniem metrum/długości (Wymagania 5.6, 5.7);
  walidacja ``target_artist`` → :class:`UnknownArtistError` (tryb ``conditional``,
  Wymaganie 5.9) i :class:`ArtistMismatchError` (tryb ``per_artist``, Wymaganie 5.10),
* :meth:`StyleTransferPipeline.infer_ga` (10.2) - inferencja ewolucyjna na
  cechach *Zbioru_Stylu* (Wymaganie 5.11),
* :meth:`StyleTransferPipeline.infer_combined` (10.3) - tryb łączony GAN → GA.

*Punkt_Kontrolny* budowany jest realnym :class:`GANTrainer` z drobną geometrią
(``conv_dim=8``, jeden blok residualny, dwie warstwy dyskryminatora) na małym
pianorollu ``[1, 16, 16]`` - inferencja używa **tej samej** geometrii, dzięki
czemu ``generator_state`` wczytuje się bez niezgodności kształtów. Testy są
szybkie (CPU, 1 epoka, mały GA).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from mido import Message, MetaMessage, MidiFile, MidiTrack

from musicians_style.config import (
    Config,
    DatasetConfig,
    GAConfig,
    GeneratorConfig,
    ModelConfig,
    TrainingConfig,
)
from musicians_style.data.manifest import FileEntry, Manifest, to_json
from musicians_style.errors import ArtistMismatchError, UnknownArtistError
from musicians_style.inference.pipeline import StyleTransferPipeline
from musicians_style.midi.parser import MidiParser
from musicians_style.midi.pianoroll import Pianoroll
from musicians_style.midi.types import InternalRepr
from musicians_style.training.dataset import MultiArtistManifest
from musicians_style.training.trainer import GANTrainer

_TICKS_PER_BEAT = 480

# Drobna geometria pianorolla - spójna między treningiem a inferencją.
_TINY_PITCH_RANGE = (48, 64)  # 16 wysokości
_TINY_STEPS_PER_BEAT = 4
_TINY_WINDOW_STEPS = 16
_STEP_TICKS = _TICKS_PER_BEAT // _TINY_STEPS_PER_BEAT  # 120 ticków na krok


# --------------------------------------------------------------------------- #
# Pomocnicze: budowa plików MIDI i manifestów
# --------------------------------------------------------------------------- #
def _write_midi(
    path: Path,
    notes: list[tuple[int, int, int]],
    *,
    numerator: int = 4,
    denominator: int = 4,
) -> None:
    """Zapisuje plik SMF (format 1) z metrum i listą nut ``(pitch, tick, dur)``."""
    mf = MidiFile(type=1, ticks_per_beat=_TICKS_PER_BEAT)
    conductor = MidiTrack()
    conductor.append(
        MetaMessage("time_signature", numerator=numerator, denominator=denominator, time=0)
    )
    mf.tracks.append(conductor)

    track = MidiTrack()
    # Buduj zdarzenia o czasach bezwzględnych, następnie konwertuj na delta.
    events: list[tuple[int, Message]] = []
    for pitch, tick, dur in notes:
        events.append((tick, Message("note_on", note=pitch, velocity=100)))
        events.append((tick + dur, Message("note_off", note=pitch, velocity=0)))
    events.sort(key=lambda e: e[0])
    prev = 0
    for abs_tick, msg in events:
        msg.time = abs_tick - prev
        prev = abs_tick
        track.append(msg)
    mf.tracks.append(track)
    mf.save(str(path))


def _make_artist_dir(
    base: Path, artist_id: str, pitch_lists: list[list[int]]
) -> tuple[Path, Manifest]:
    """Tworzy katalog artysty z plikami MIDI i odpowiadający Manifest."""
    directory = base / artist_id
    directory.mkdir(parents=True, exist_ok=True)
    entries: list[FileEntry] = []
    for i, pitches in enumerate(pitch_lists):
        name = f"track_{i:02d}.mid"
        notes = [(p, j * _TICKS_PER_BEAT, _TICKS_PER_BEAT) for j, p in enumerate(pitches)]
        _write_midi(directory / name, notes)
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


def _tiny_pianoroll() -> Pianoroll:
    return Pianoroll(
        pitch_range=_TINY_PITCH_RANGE,
        steps_per_beat=_TINY_STEPS_PER_BEAT,
        window_steps=_TINY_WINDOW_STEPS,
    )


def _tiny_config(mode: str, *, artists: list[str], reference_artist: str | None = None) -> Config:
    return Config(
        seed=0,
        experiment_name="inference_unit_test",
        dataset=DatasetConfig(min_files_per_artist=1),
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


def _build_conditional_checkpoint(tmp_path: Path) -> tuple[Path, list[str]]:
    """Trenuje drobny StarGAN (1 epoka) i zwraca ścieżkę Punktu_Kontrolnego."""
    dir_a, m_a = _make_artist_dir(tmp_path / "train", "alpha", [[50, 55], [52, 57]])
    dir_b, m_b = _make_artist_dir(tmp_path / "train", "bravo", [[54, 59], [49, 53]])
    multi = MultiArtistManifest([m_a, m_b])

    config = _tiny_config("conditional", artists=["alpha", "bravo"])
    trainer = GANTrainer(
        config,
        "conditional",
        output_dir=tmp_path / "exp_cond",
        roots={"alpha": dir_a, "bravo": dir_b},
        pianoroll=_tiny_pianoroll(),
        conv_dim=8,
        n_residual_blocks=1,
        disc_layers=2,
    )
    checkpoint_path = trainer.train(multi, seed=123)
    return checkpoint_path, ["alpha", "bravo"]


def _build_per_artist_checkpoint(tmp_path: Path) -> Path:
    """Trenuje drobny CycleGAN (1 epoka) i zwraca ścieżkę Punktu_Kontrolnego."""
    dir_solo, m_solo = _make_artist_dir(tmp_path / "train", "solo", [[50, 55], [52, 57]])
    config = _tiny_config("per_artist", artists=["solo"], reference_artist="control")
    trainer = GANTrainer(
        config,
        "per_artist",
        output_dir=tmp_path / "exp_solo",
        roots=dir_solo,
        pianoroll=_tiny_pianoroll(),
        conv_dim=8,
        n_residual_blocks=1,
        disc_layers=2,
    )
    return trainer.train(m_solo, seed=42)


def _make_pipeline(**kwargs) -> StyleTransferPipeline:
    """Buduje pipeline z drobną geometrią spójną z Punktem_Kontrolnym."""
    return StyleTransferPipeline(
        pianoroll=_tiny_pianoroll(),
        conv_dim=8,
        n_residual_blocks=1,
        ga_config=_small_ga_config(),
        device="cpu",
        **kwargs,
    )


def _small_ga_config() -> GAConfig:
    """Mały, szybki *Algorytm_Genetyczny* do testów (kilka osobników/pokoleń)."""
    return GAConfig(
        population_size=4,
        generations=2,
        tournament_size=2,
        crossover="uniform",
        elitism_k=1,
        fitness_metric="euclidean",
        stagnation_generations=2,
    )


def _write_input_midi(path: Path, *, numerator: int = 3, denominator: int = 4) -> None:
    """Zapisuje *Utwór_Wejściowy* z metrum 3/4 i nutami w siatce pianorolla."""
    notes = [(50, 0, _TICKS_PER_BEAT), (55, _TICKS_PER_BEAT, _TICKS_PER_BEAT)]
    _write_midi(path, notes, numerator=numerator, denominator=denominator)


def _time_signature(repr_: InternalRepr) -> tuple[int, int] | None:
    """Zwraca ``(numerator, denominator)`` pierwszego metrum lub ``None``."""
    for meta in repr_.meta:
        if meta.kind == "time_signature":
            return int(meta.payload["numerator"]), int(meta.payload["denominator"])
    return None


def _total_length_ticks(repr_: InternalRepr) -> int:
    """Długość utworu w tickach: ``max(tick + duration_ticks)`` (0 gdy brak nut)."""
    if not repr_.notes:
        return 0
    return max(note.tick + note.duration_ticks for note in repr_.notes)


class _IdentityGenerator(nn.Module):
    """Generator tożsamościowy ``G(x[, c]) = x`` (deterministyczna kontrola długości)."""

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:  # noqa: D102
        return x


# --------------------------------------------------------------------------- #
# 10.1 - infer_gan: happy path
# --------------------------------------------------------------------------- #
def test_infer_gan_creates_valid_midi_and_preserves_meter(tmp_path: Path) -> None:
    checkpoint_path, _ = _build_conditional_checkpoint(tmp_path)
    input_path = tmp_path / "input.mid"
    _write_input_midi(input_path, numerator=3, denominator=4)

    pipeline = _make_pipeline()
    output_path = pipeline.infer_gan(
        input_path, target_artist="alpha", checkpoint_path=checkpoint_path, seed=7
    )

    # Plik wyjściowy powstał.
    assert output_path.exists()

    parser = MidiParser()
    # Poprawny SMF: walidacja struktury nie zgłasza wyjątku (Wymaganie 5.2).
    parser.validate(output_path)
    output_repr = parser.parse(output_path)
    input_repr = parser.parse(input_path)

    # Metrum zachowane (Wymaganie 5.7) - meta pochodzi z template = wejście.
    assert _time_signature(output_repr) == _time_signature(input_repr) == (3, 4)
    # Siatka czasowa i format zachowane z template.
    assert output_repr.ticks_per_beat == input_repr.ticks_per_beat
    assert output_repr.smf_format == input_repr.smf_format


def test_infer_gan_identity_generator_preserves_length(tmp_path: Path) -> None:
    # Z generatorem tożsamościowym pianoroll wejścia jest odtwarzany 1:1,
    # więc długość Utworu_Wyjściowego równa się długości wejścia (Wymaganie 5.6).
    checkpoint_path, _ = _build_conditional_checkpoint(tmp_path)
    input_path = tmp_path / "input.mid"
    _write_input_midi(input_path)

    pipeline = _make_pipeline()
    output_path = pipeline.infer_gan(
        input_path,
        target_artist="alpha",
        checkpoint_path=checkpoint_path,
        seed=1,
        generator=_IdentityGenerator(),
    )

    parser = MidiParser()
    input_repr = parser.parse(input_path)
    output_repr = parser.parse(output_path)

    in_len = _total_length_ticks(input_repr)
    out_len = _total_length_ticks(output_repr)
    assert in_len > 0 and out_len > 0
    ratio = out_len / in_len
    assert 0.95 <= ratio <= 1.05  # Property 11 / Wymaganie 5.6
    assert _time_signature(output_repr) == _time_signature(input_repr)


def test_infer_gan_default_output_path(tmp_path: Path) -> None:
    checkpoint_path, _ = _build_conditional_checkpoint(tmp_path)
    input_path = tmp_path / "song.mid"
    _write_input_midi(input_path)

    pipeline = _make_pipeline()
    output_path = pipeline.infer_gan(
        input_path, target_artist="alpha", checkpoint_path=checkpoint_path
    )
    # Domyślna nazwa: <stem>_gan.mid obok wejścia.
    assert output_path == input_path.parent / "song_gan.mid"
    assert output_path.exists()


# --------------------------------------------------------------------------- #
# 10.1 - walidacja target_artist (Wymagania 5.9, 5.10)
# --------------------------------------------------------------------------- #
def test_infer_gan_unknown_artist_conditional(tmp_path: Path) -> None:
    checkpoint_path, artists = _build_conditional_checkpoint(tmp_path)
    input_path = tmp_path / "input.mid"
    _write_input_midi(input_path)

    pipeline = _make_pipeline()
    with pytest.raises(UnknownArtistError) as exc_info:
        pipeline.infer_gan(
            input_path, target_artist="nieznany", checkpoint_path=checkpoint_path
        )
    # Błąd niesie listę dostępnych artystów i kod wyjścia 2 (Wymaganie 5.9).
    assert exc_info.value.available == artists
    assert exc_info.value.exit_code == 2
    # Inferencja nie została wykonana - brak pliku wyjściowego.
    assert not (input_path.parent / "input_gan.mid").exists()


def test_infer_gan_artist_mismatch_per_artist(tmp_path: Path) -> None:
    checkpoint_path = _build_per_artist_checkpoint(tmp_path)
    input_path = tmp_path / "input.mid"
    _write_input_midi(input_path)

    pipeline = _make_pipeline()
    with pytest.raises(ArtistMismatchError) as exc_info:
        pipeline.infer_gan(
            input_path, target_artist="ktokolwiek", checkpoint_path=checkpoint_path
        )
    assert exc_info.value.expected == "solo"
    assert exc_info.value.exit_code == 2


def test_infer_gan_per_artist_matching_artist_succeeds(tmp_path: Path) -> None:
    checkpoint_path = _build_per_artist_checkpoint(tmp_path)
    input_path = tmp_path / "input.mid"
    _write_input_midi(input_path)

    pipeline = _make_pipeline()
    output_path = pipeline.infer_gan(
        input_path, target_artist="solo", checkpoint_path=checkpoint_path, seed=3
    )
    assert output_path.exists()
    MidiParser().validate(output_path)


# --------------------------------------------------------------------------- #
# 10.2 - infer_ga
# --------------------------------------------------------------------------- #
def _build_style_manifest(tmp_path: Path, artist_id: str = "alpha") -> Path:
    """Tworzy *Zbiór_Stylu* (MIDI + manifest JSON) i zwraca ścieżkę manifestu."""
    directory, manifest = _make_artist_dir(
        tmp_path / "style", artist_id, [[50, 53, 57], [52, 55, 59]]
    )
    manifest_path = directory / "manifest.json"
    to_json(manifest, manifest_path)
    return manifest_path


def test_infer_ga_creates_valid_midi(tmp_path: Path) -> None:
    manifest_path = _build_style_manifest(tmp_path, "alpha")
    input_path = tmp_path / "input.mid"
    _write_input_midi(input_path, numerator=3, denominator=4)

    pipeline = _make_pipeline()
    output_path = pipeline.infer_ga(
        input_path,
        target_artist="alpha",
        style_manifest_path=manifest_path,
        seed=11,
    )

    assert output_path.exists()
    parser = MidiParser()
    parser.validate(output_path)
    output_repr = parser.parse(output_path)
    input_repr = parser.parse(input_path)
    # GA aplikuje apply_transformation - meta (metrum) zachowane (Wymaganie 5.7).
    assert _time_signature(output_repr) == _time_signature(input_repr) == (3, 4)


def test_infer_ga_unknown_artist(tmp_path: Path) -> None:
    manifest_path = _build_style_manifest(tmp_path, "alpha")
    input_path = tmp_path / "input.mid"
    _write_input_midi(input_path)

    pipeline = _make_pipeline()
    with pytest.raises(UnknownArtistError):
        pipeline.infer_ga(
            input_path, target_artist="inny", style_manifest_path=manifest_path
        )


def test_infer_ga_writes_history_log(tmp_path: Path) -> None:
    manifest_path = _build_style_manifest(tmp_path, "alpha")
    input_path = tmp_path / "input.mid"
    _write_input_midi(input_path)
    log_path = tmp_path / "ga.jsonl"

    pipeline = _make_pipeline()
    pipeline.infer_ga(
        input_path,
        target_artist="alpha",
        style_manifest_path=manifest_path,
        seed=5,
        log_path=log_path,
    )
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8").strip()  # niepusty log pokoleń


# --------------------------------------------------------------------------- #
# 10.3 - infer_combined
# --------------------------------------------------------------------------- #
def test_infer_combined_gan_then_ga(tmp_path: Path) -> None:
    checkpoint_path, _ = _build_conditional_checkpoint(tmp_path)
    manifest_path = _build_style_manifest(tmp_path, "alpha")
    input_path = tmp_path / "input.mid"
    _write_input_midi(input_path, numerator=3, denominator=4)

    pipeline = _make_pipeline()
    output_path = pipeline.infer_combined(
        input_path,
        target_artist="alpha",
        checkpoint_path=checkpoint_path,
        style_manifest_path=manifest_path,
        seed=9,
        order="gan_then_ga",
    )

    assert output_path == input_path.parent / "input_combined.mid"
    assert output_path.exists()
    parser = MidiParser()
    parser.validate(output_path)
    output_repr = parser.parse(output_path)
    # Metrum przenoszone przez cały łańcuch GAN → GA (Wymaganie 5.7).
    assert _time_signature(output_repr) == (3, 4)


def test_infer_combined_ga_then_gan(tmp_path: Path) -> None:
    checkpoint_path, _ = _build_conditional_checkpoint(tmp_path)
    manifest_path = _build_style_manifest(tmp_path, "alpha")
    input_path = tmp_path / "input.mid"
    _write_input_midi(input_path)

    pipeline = _make_pipeline()
    output_path = pipeline.infer_combined(
        input_path,
        target_artist="alpha",
        checkpoint_path=checkpoint_path,
        style_manifest_path=manifest_path,
        seed=9,
        order="ga_then_gan",
        generator=_IdentityGenerator(),
    )
    assert output_path.exists()
    MidiParser().validate(output_path)


def test_infer_combined_invalid_order(tmp_path: Path) -> None:
    checkpoint_path, _ = _build_conditional_checkpoint(tmp_path)
    manifest_path = _build_style_manifest(tmp_path, "alpha")
    input_path = tmp_path / "input.mid"
    _write_input_midi(input_path)

    pipeline = _make_pipeline()
    with pytest.raises(ValueError):
        pipeline.infer_combined(
            input_path,
            target_artist="alpha",
            checkpoint_path=checkpoint_path,
            style_manifest_path=manifest_path,
            order="bogus",  # type: ignore[arg-type]
        )


def test_infer_combined_validates_target_artist(tmp_path: Path) -> None:
    checkpoint_path, _ = _build_conditional_checkpoint(tmp_path)
    manifest_path = _build_style_manifest(tmp_path, "alpha")
    input_path = tmp_path / "input.mid"
    _write_input_midi(input_path)

    pipeline = _make_pipeline()
    with pytest.raises(UnknownArtistError):
        pipeline.infer_combined(
            input_path,
            target_artist="nieznany",
            checkpoint_path=checkpoint_path,
            style_manifest_path=manifest_path,
        )
