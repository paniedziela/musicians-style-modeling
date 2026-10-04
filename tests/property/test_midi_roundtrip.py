# Feature: musicians-style-modeling, Property 1
"""Property 1: Round-trip parsera i pretty printera MIDI (zadanie 2.4).

**Validates: Requirements 5.4, 11.1**

Sekcja *Correctness Properties* (``design.md``):

    *For any* poprawnej *Reprezentacji_Wewnętrznej* ``r``, sekwencja operacji
    ``parse(write(r))`` produkuje *Reprezentację_Wewnętrzną* ``r'`` semantycznie
    równoważną ``r``, to znaczy zawierającą tę samą listę zdarzeń nutowych z
    dokładnością do kolejności zdarzeń o identycznym znaczniku czasu.

Wymaganie 5.4 (``requirements.md``): gdy parser wczyta poprawny plik, a następnie
pretty printer zapisze plik wynikowy i parser ponownie go wczyta, System MA
produkować reprezentację wewnętrzną semantycznie równoważną oryginalnej (ta sama
lista zdarzeń nutowych z dokładnością do kolejności zdarzeń o identycznym
znaczniku czasu). Wymaganie 11.1 podnosi ten warunek do rangi testu własnościowego.

Strategia ``midi_internal_repr`` pochodzi ze wspólnego, przetestowanego modułu
``tests/property/strategies.py`` i z założenia generuje *Reprezentacje_Wewnętrzne*
round-trippowalne (``velocity >= 1``, brak nakładających się nut o tej samej parze
``(channel, pitch)``, deduplikacja meta-zdarzeń), więc właściwość powinna
zachodzić bez dodatkowych założeń.
"""

from __future__ import annotations

import pytest
from .profiles import property_settings
from hypothesis import given

from musicians_style.midi.parser import MidiParser
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.midi.types import InternalRepr, NoteEvent

from .strategies import midi_internal_repr


def _note_signature(note: NoteEvent) -> tuple[int, int, int, int, int]:
    """Porównywalna krotka zdarzenia nutowego ``(tick, channel, pitch, velocity, duration)``.

    Semantyczna równoważność (Wymaganie 5.4) dotyczy *zbioru/multizbioru* zdarzeń
    nutowych z dokładnością do kolejności zdarzeń o identycznym znaczniku czasu.
    Sprowadzamy zatem każdą nutę do krotki jej istotnych pól i porównujemy
    posortowane listy takich krotek, co jest niewrażliwe na kolejność.
    """
    return (
        note.tick,
        note.channel,
        note.pitch,
        note.velocity,
        note.duration_ticks,
    )


def _sorted_note_signatures(repr_: InternalRepr) -> list[tuple[int, int, int, int, int]]:
    """Zwraca posortowaną listę krotek zdarzeń nutowych z *Reprezentacji_Wewnętrznej*."""
    return sorted(_note_signature(note) for note in repr_.notes)


@pytest.mark.property
@property_settings(max_examples=200, deadline=None)
@given(repr_=midi_internal_repr())
def test_parse_write_roundtrip_preserves_notes(repr_: InternalRepr) -> None:
    """``parse(write(r))`` zachowuje multizbiór zdarzeń nutowych (Property 1).

    Zapisujemy *Reprezentację_Wewnętrzną* do bajtów pretty printerem, ponownie
    parsujemy i sprawdzamy, że posortowane listy krotek
    ``(tick, channel, pitch, velocity, duration_ticks)`` są identyczne -
    semantyczna równoważność z dokładnością do kolejności zdarzeń o tym samym
    znaczniku czasu (Wymaganie 5.4, 11.1).
    """
    printer = MidiPrettyPrinter()
    parser = MidiParser()

    serialized = printer.to_bytes(repr_)
    reparsed = parser.parse_bytes(serialized)

    assert _sorted_note_signatures(reparsed) == _sorted_note_signatures(repr_)
