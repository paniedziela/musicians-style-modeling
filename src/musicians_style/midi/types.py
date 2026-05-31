"""Reprezentacja_Wewnętrzna plików MIDI (Wymaganie 5.3).

Moduł definiuje niemutowalne struktury danych stanowiące *Reprezentację_Wewnętrzną*
plików MIDI, używaną przez ``Parser_MIDI`` i ``Pretty_Printer_MIDI`` (sekcja
*Components and Interfaces* w ``design.md``).

Kluczowe założenia projektowe:

- Wszystkie struktury są ``@dataclass(frozen=True)`` - niemutowalne, porównywalne
  po wartości i haszowalne. Dzięki temu nadają się do testów własnościowych
  (property-based testing, Wymaganie 11) oraz do użycia jako klucze w zbiorach/
  słownikach przy weryfikacji round-trip (Wymaganie 5.4, 11.1).
- Kolekcje (``notes``, ``meta``) są przechowywane jako krotki, nie listy, aby
  zachować pełną niemutowalność i haszowalność ``InternalRepr``.
- Sortowanie zdarzeń o identycznym znaczniku czasu jest deterministyczne dzięki
  funkcji :func:`event_key` (lex po ``(tick, channel, pitch, velocity)``), co
  spełnia wymóg "semantycznej równoważności z dokładnością do kolejności zdarzeń
  o identycznym znaczniku czasu" (Wymaganie 5.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "NoteEvent",
    "MetaEvent",
    "InternalRepr",
    "MetaKind",
    "event_key",
]

# Dozwolone rodzaje zdarzeń meta utrzymywanych w Reprezentacji_Wewnętrznej.
MetaKind = Literal["tempo", "time_signature", "key_signature", "program_change"]


@dataclass(frozen=True)
class NoteEvent:
    """Pojedyncze zdarzenie nutowe w *Reprezentacji_Wewnętrznej*.

    Atrybuty (zgodnie z sekcją *Parser_MIDI* w ``design.md``):
        tick: czas rozpoczęcia zdarzenia w tickach SMF (>= 0).
        channel: kanał MIDI w zakresie 0..15.
        pitch: wysokość dźwięku w zakresie 0..127.
        velocity: dynamika w zakresie 0..127; wartość 0 oznacza *note-off*.
        duration_ticks: długość trwania nuty w tickach SMF (>= 0).

    Instancje są niemutowalne i haszowalne (wszystkie pola to ``int``), dzięki
    czemu mogą być elementami krotki ``InternalRepr.notes`` oraz uczestniczyć
    w porównaniach wartościowych w testach własnościowych.
    """

    tick: int
    channel: int
    pitch: int
    velocity: int
    duration_ticks: int


def _freeze_for_hash(value: Any) -> Any:
    """Zamienia strukturę na haszowalną reprezentację (na potrzeby ``__hash__``).

    ``MetaEvent.payload`` jest słownikiem (``dict[str, Any]``), który nie jest
    haszowalny. Aby ``MetaEvent`` - a w konsekwencji ``InternalRepr`` - pozostał
    haszowalny, podczas obliczania skrótu spłaszczamy słowniki i listy do krotek.
    Równość (``__eq__``) nadal porównuje oryginalne wartości pól.
    """

    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_for_hash(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_hash(v) for v in value)
    return value


@dataclass(frozen=True)
class MetaEvent:
    """Zdarzenie meta (tempo, metrum, tonacja, zmiana programu).

    Atrybuty (zgodnie z sekcją *Parser_MIDI* w ``design.md``):
        tick: czas zdarzenia w tickach SMF (>= 0).
        kind: rodzaj zdarzenia meta z dozwolonego zbioru :data:`MetaKind`.
        payload: słownik parametrów zdarzenia (np. ``{"bpm": 120.0}`` dla tempa,
            ``{"numerator": 4, "denominator": 4}`` dla metrum).

    Ponieważ ``payload`` jest słownikiem, automatycznie generowany ``__hash__``
    klasy ``@dataclass(frozen=True)`` zgłosiłby ``TypeError`` przy haszowaniu.
    Definiujemy więc własny ``__hash__`` (dataclass zachowuje jawnie zdefiniowaną
    metodę), tak aby ``MetaEvent`` był haszowalny - co jest potrzebne dla
    haszowalności całej ``InternalRepr``.
    """

    tick: int
    kind: MetaKind
    payload: dict[str, Any]

    def __hash__(self) -> int:  # noqa: D105 - opisane w docstringu klasy
        return hash((self.tick, self.kind, _freeze_for_hash(self.payload)))


@dataclass(frozen=True)
class InternalRepr:
    """*Reprezentacja_Wewnętrzna* pojedynczego pliku MIDI.

    Atrybuty (zgodnie z sekcją *Parser_MIDI* w ``design.md``):
        ticks_per_beat: rozdzielczość czasowa SMF (ticki na ćwierćnutę, > 0).
        notes: krotka zdarzeń nutowych, posortowana deterministycznie wg
            :func:`event_key` (lex po ``(tick, channel, pitch, velocity)``).
        meta: krotka zdarzeń meta (tempo, metrum, tonacja, zmiana programu).
        smf_format: format Standard MIDI File - 0 lub 1.

    Kolekcje są krotkami, dzięki czemu cała struktura jest niemutowalna,
    porównywalna po wartości i haszowalna (Wymaganie 11 - testy własnościowe).
    """

    ticks_per_beat: int
    notes: tuple[NoteEvent, ...] = field(default_factory=tuple)
    meta: tuple[MetaEvent, ...] = field(default_factory=tuple)
    smf_format: int = 1


def event_key(e: NoteEvent) -> tuple[int, int, int, int]:
    """Klucz deterministycznego, leksykograficznego sortowania zdarzeń nutowych.

    Zwraca krotkę ``(tick, channel, pitch, velocity)``. Użycie tego klucza przy
    sortowaniu ``notes`` zapewnia, że zdarzenia o identycznym znaczniku czasu
    (``tick``) mają zawsze tę samą, powtarzalną kolejność - warunek konieczny do
    spełnienia własności round-trip (Wymaganie 5.4, 11.1).

    Args:
        e: zdarzenie nutowe ``NoteEvent``.

    Returns:
        Krotka ``(tick, channel, pitch, velocity)`` do porównań leksykograficznych.
    """

    return (e.tick, e.channel, e.pitch, e.velocity)
