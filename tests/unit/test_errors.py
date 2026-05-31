"""Testy jednostkowe hierarchii wyjątków (``musicians_style.errors``).

Weryfikują niezmienniki opisane w sekcji *Error Handling* dokumentu
``design.md``: niepusty opis :class:`MidiValidationError` (Wymaganie 5.5),
przenoszenie kontekstu przez :class:`UnknownArtistError` /
:class:`ArtistMismatchError` (Wymagania 5.9, 5.10), kody wyjścia CLI oraz
wspólny przodek :class:`MusiciansStyleError`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from musicians_style.errors import (
    MIDI_VALIDATION_DEFAULT_MESSAGE,
    ArtistMismatchError,
    ConfigValidationError,
    EmptyDatasetError,
    GpuOutOfMemoryError,
    MidiValidationError,
    MusiciansStyleError,
    UnknownArtistError,
)

ALL_ERRORS = [
    MidiValidationError,
    EmptyDatasetError,
    UnknownArtistError,
    ArtistMismatchError,
    GpuOutOfMemoryError,
    ConfigValidationError,
]


@pytest.mark.parametrize("error_cls", ALL_ERRORS)
def test_all_errors_inherit_base(error_cls: type) -> None:
    assert issubclass(error_cls, MusiciansStyleError)
    assert issubclass(error_cls, Exception)


def test_midi_validation_error_preserves_message() -> None:
    err = MidiValidationError("brak nagłówka MThd")
    assert err.message == "brak nagłówka MThd"
    assert "brak nagłówka MThd" in str(err)


def test_midi_validation_error_defaults_when_empty() -> None:
    # Pusty łańcuch i None muszą skutkować niepustym, domyślnym opisem.
    for empty in ("", None):
        err = MidiValidationError(empty)  # type: ignore[arg-type]
        assert err.message == MIDI_VALIDATION_DEFAULT_MESSAGE
        assert err.message  # niepusty


def test_midi_validation_error_includes_location() -> None:
    err = MidiValidationError("zła długość chunku", file=Path("broken.mid"), offset=14)
    assert err.file == Path("broken.mid")
    assert err.offset == 14
    text = str(err)
    assert "broken.mid" in text
    assert "14" in text


def test_unknown_artist_error_carries_available_list() -> None:
    err = UnknownArtistError("nirvana", available=["beatles", "queen"])
    assert err.requested == "nirvana"
    assert err.available == ["beatles", "queen"]
    assert "nirvana" in str(err)
    assert "beatles" in str(err)


def test_artist_mismatch_error_carries_expected() -> None:
    err = ArtistMismatchError("queen", expected="beatles")
    assert err.requested == "queen"
    assert err.expected == "beatles"
    assert "beatles" in str(err)


def test_gpu_oom_error_suggests_batch_size() -> None:
    err = GpuOutOfMemoryError(suggested_batch_size=8)
    assert err.suggested_batch_size == 8
    assert "8" in str(err)


@pytest.mark.parametrize(
    ("error", "expected_exit"),
    [
        (MidiValidationError("x"), 2),
        (EmptyDatasetError("brak plików"), 2),
        (UnknownArtistError("a", ["b"]), 2),
        (ArtistMismatchError("a", "b"), 2),
        (ConfigValidationError("brak seed"), 2),
        (GpuOutOfMemoryError(), 3),
    ],
)
def test_exit_codes(error: MusiciansStyleError, expected_exit: int) -> None:
    assert error.exit_code == expected_exit


def test_base_error_default_exit_code() -> None:
    assert MusiciansStyleError().exit_code == 1
