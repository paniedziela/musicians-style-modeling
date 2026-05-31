"""Testy jednostkowe :class:`musicians_style.midi.parser.MidiParser` (zadanie 2.2).

Weryfikują kontrakt parsera z sekcji *Parser_MIDI* (``design.md``) oraz
Wymagania 5.3 i 5.5:

* konwersję pliku SMF (format 0 i 1) na *Reprezentację_Wewnętrzną* z
  deterministycznie posortowanymi krotkami zdarzeń,
* round-trip z :class:`MidiPrettyPrinter` (semantyczna równoważność nut),
* walidację struktury na surowych bajtach **przed** przetwarzaniem treści,
  z niepustym opisem błędu i bez nieobsłużonych wyjątków.
"""

from __future__ import annotations

import io

import pytest
from mido import MidiFile, MidiTrack, Message, MetaMessage

from musicians_style.errors import MidiValidationError
from musicians_style.midi.parser import MidiParser
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent, event_key


@pytest.fixture()
def parser() -> MidiParser:
    return MidiParser()


@pytest.fixture()
def printer() -> MidiPrettyPrinter:
    return MidiPrettyPrinter()


def _sample_repr(smf_format: int) -> InternalRepr:
    notes = (
        NoteEvent(tick=0, channel=0, pitch=60, velocity=100, duration_ticks=480),
        NoteEvent(tick=480, channel=0, pitch=64, velocity=90, duration_ticks=240),
        NoteEvent(tick=480, channel=1, pitch=67, velocity=80, duration_ticks=240),
    )
    meta = (
        MetaEvent(tick=0, kind="tempo", payload={"tempo": 500_000}),
        MetaEvent(tick=0, kind="time_signature", payload={"numerator": 3, "denominator": 4}),
        MetaEvent(tick=0, kind="key_signature", payload={"key": "C"}),
        MetaEvent(tick=0, kind="program_change", payload={"program": 1, "channel": 0}),
    )
    return InternalRepr(ticks_per_beat=480, notes=notes, meta=meta, smf_format=smf_format)


# -- round-trip / konwersja treści ------------------------------------------


@pytest.mark.parametrize("smf_format", [0, 1])
def test_roundtrip_preserves_notes(
    parser: MidiParser, printer: MidiPrettyPrinter, smf_format: int
) -> None:
    original = _sample_repr(smf_format)
    parsed = parser.parse_bytes(printer.to_bytes(original))
    assert parsed.notes == original.notes
    assert parsed.ticks_per_beat == original.ticks_per_beat
    assert parsed.smf_format == smf_format


@pytest.mark.parametrize("smf_format", [0, 1])
def test_roundtrip_preserves_meta_kinds(
    parser: MidiParser, printer: MidiPrettyPrinter, smf_format: int
) -> None:
    original = _sample_repr(smf_format)
    parsed = parser.parse_bytes(printer.to_bytes(original))
    assert sorted(m.kind for m in parsed.meta) == sorted(m.kind for m in original.meta)
    tempo = next(m for m in parsed.meta if m.kind == "tempo")
    assert tempo.payload["tempo"] == 500_000
    timesig = next(m for m in parsed.meta if m.kind == "time_signature")
    assert (timesig.payload["numerator"], timesig.payload["denominator"]) == (3, 4)


def test_parse_bytes_returns_sorted_notes(
    parser: MidiParser, printer: MidiPrettyPrinter
) -> None:
    parsed = parser.parse_bytes(printer.to_bytes(_sample_repr(1)))
    keys = [event_key(n) for n in parsed.notes]
    assert keys == sorted(keys)


def test_single_note_file(parser: MidiParser, printer: MidiPrettyPrinter) -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(NoteEvent(tick=0, channel=0, pitch=72, velocity=64, duration_ticks=480),),
        smf_format=1,
    )
    parsed = parser.parse_bytes(printer.to_bytes(repr_))
    assert parsed.notes == repr_.notes


def test_parse_from_path(tmp_path, parser: MidiParser, printer: MidiPrettyPrinter) -> None:
    path = tmp_path / "sample.mid"
    printer.write(_sample_repr(1), path)
    parsed_from_path = parser.parse(path)
    parsed_from_bytes = parser.parse_bytes(path.read_bytes())
    assert parsed_from_path == parsed_from_bytes


def test_note_on_zero_velocity_is_note_off(parser: MidiParser) -> None:
    mf = MidiFile(type=0, ticks_per_beat=480)
    track = MidiTrack()
    track.append(Message("note_on", note=60, velocity=100, time=0))
    track.append(Message("note_on", note=60, velocity=0, time=240))
    mf.tracks.append(track)
    buffer = io.BytesIO()
    mf.save(file=buffer)
    parsed = parser.parse_bytes(buffer.getvalue())
    assert len(parsed.notes) == 1
    assert parsed.notes[0].duration_ticks == 240


def test_file_without_tempo_has_no_tempo_meta(parser: MidiParser) -> None:
    # Domyślne 120 BPM jest obowiązkiem Ekstraktora_Cech (Wymaganie 2.2),
    # nie parsera - parser nie wstrzykuje sztucznego tempa.
    mf = MidiFile(type=1, ticks_per_beat=480)
    conductor = MidiTrack()
    conductor.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    mf.tracks.append(conductor)
    notes = MidiTrack()
    notes.append(Message("note_on", note=60, velocity=100, time=0))
    notes.append(Message("note_off", note=60, velocity=0, time=480))
    mf.tracks.append(notes)
    buffer = io.BytesIO()
    mf.save(file=buffer)
    parsed = parser.parse_bytes(buffer.getvalue())
    assert [m for m in parsed.meta if m.kind == "tempo"] == []
    assert len(parsed.notes) == 1


def test_key_and_time_signature_parsed(parser: MidiParser) -> None:
    mf = MidiFile(type=1, ticks_per_beat=480)
    conductor = MidiTrack()
    conductor.append(MetaMessage("key_signature", key="Am", time=0))
    conductor.append(MetaMessage("time_signature", numerator=6, denominator=8, time=0))
    mf.tracks.append(conductor)
    mf.tracks.append(MidiTrack())
    buffer = io.BytesIO()
    mf.save(file=buffer)
    parsed = parser.parse_bytes(buffer.getvalue())
    key_meta = next(m for m in parsed.meta if m.kind == "key_signature")
    assert key_meta.payload["key"] == "Am"
    ts_meta = next(m for m in parsed.meta if m.kind == "time_signature")
    assert (ts_meta.payload["numerator"], ts_meta.payload["denominator"]) == (6, 8)


# -- walidacja struktury (surowe bajty, Wymaganie 5.5) ----------------------


def test_validate_accepts_valid_bytes(
    parser: MidiParser, printer: MidiPrettyPrinter
) -> None:
    assert parser.validate(printer.to_bytes(_sample_repr(1))) is None


def test_validate_accepts_valid_path(
    tmp_path, parser: MidiParser, printer: MidiPrettyPrinter
) -> None:
    path = tmp_path / "ok.mid"
    printer.write(_sample_repr(0), path)
    assert parser.validate(path) is None


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("empty", b""),
        ("short_header", b"MThd"),
        ("truncated_body", b"MThd\x00\x00\x00\x06\x00\x01"),
        ("bad_signature", b"XXXX\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0"),
        (
            "short_header_len",
            b"MThd\x00\x00\x00\x04\x00\x00\x00\x01",
        ),
        (
            "format0_two_tracks",
            b"MThd\x00\x00\x00\x06\x00\x00\x00\x02\x01\xe0",
        ),
        (
            "wrong_chunk_id",
            b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0XXXX\x00\x00\x00\x00",
        ),
        (
            "track_len_overflow",
            b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0MTrk\x00\x00\xff\xff",
        ),
        (
            "track_count_mismatch",
            b"MThd\x00\x00\x00\x06\x00\x01\x00\x03\x01\xe0",
        ),
    ],
)
def test_validate_rejects_corrupt_with_message(
    parser: MidiParser, name: str, data: bytes
) -> None:
    with pytest.raises(MidiValidationError) as excinfo:
        parser.validate(data)
    assert excinfo.value.message  # niepusty opis (Wymaganie 5.5)


def test_parse_corrupt_raises_validation_error(parser: MidiParser) -> None:
    with pytest.raises(MidiValidationError):
        parser.parse_bytes(b"not a midi file at all")


def test_validate_never_raises_unhandled_for_random_bytes(parser: MidiParser) -> None:
    # Property 13 (smoke): dowolne bajty -> sukces lub MidiValidationError.
    import os

    for size in range(0, 50):
        data = os.urandom(size)
        try:
            parser.validate(data)
        except MidiValidationError as exc:
            assert exc.message
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"nieobsłużony wyjątek {type(exc).__name__} dla {data!r}: {exc}")
