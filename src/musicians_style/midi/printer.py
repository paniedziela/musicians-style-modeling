"""Pretty_Printer_MIDI - serializacja *Reprezentacji_Wewnętrznej* do SMF (Wymagania 5.2, 5.3).

Moduł implementuje :class:`MidiPrettyPrinter` - komponent odwrotny do
``Parser_MIDI`` (``parser.py``). Przekształca :class:`~musicians_style.midi.types.InternalRepr`
z powrotem na plik *Standard MIDI File* zgodny ze specyfikacją MIDI 1.0
(format 0 lub format 1, w zależności od ``InternalRepr.smf_format``).

Komponent jest centralny dla własności round-trip (Property 1, Wymaganie 5.4,
11.1): dla poprawnej *Reprezentacji_Wewnętrznej* ``r`` sekwencja
``parse(write(r))`` ma produkować ``r'`` semantycznie równoważne ``r`` - tę
samą listę zdarzeń nutowych z dokładnością do kolejności zdarzeń o identycznym
znaczniku czasu.

Schemat payloadów meta-zdarzeń (kontrakt z Parser_MIDI)
-------------------------------------------------------
Aby printer był odwrotnością parsera, oba komponenty muszą uzgodnić schemat
słownika ``MetaEvent.payload``. Parser (``parser.py``, zadanie 2.2) jest
budowany równolegle - jeśli definiuje własny schemat w docstringu modułu,
**ma on pierwszeństwo** i printer należy z nim zsynchronizować. Dopóki parser
nie istnieje, obowiązuje poniższy kanoniczny, bezpieczny dla round-trip schemat:

* ``tempo``           → ``{"tempo": <mikrosekundy na ćwierćnutę>}``
  (mikrosekundy, nie BPM - mido przechowuje tempo jako 24-bitową liczbę
  całkowitą mikrosekund, więc unikamy stratnej konwersji BPM↔µs.
  Dla zgodności wstecznej akceptowany jest również klucz ``{"bpm": <float>}``,
  konwertowany przez ``round(60_000_000 / bpm)``).
* ``time_signature``  → ``{"numerator": <int>, "denominator": <int potęga 2>}``.
* ``key_signature``   → ``{"key": "<nazwa tonacji mido>"}`` (np. ``"C"``, ``"Am"``,
  ``"F#"``); wartość przekazywana jest do ``mido`` bez modyfikacji.
* ``program_change``  → ``{"program": <0..127>, "channel": <0..15>}``.

Kolejność zdarzeń o identycznym znaczniku czasu
-----------------------------------------------
W obrębie pojedynczej ścieżki, dla zdarzeń o tym samym znaczniku czasu (``tick``)
stosowana jest deterministyczna kolejność (od najwyższego priorytetu):

1. meta-zdarzenia (``set_tempo`` → ``time_signature`` → ``key_signature``),
2. ``program_change`` (ustawienie instrumentu przed dźwiękiem),
3. ``note_off`` nut o dodatniej długości (zwolnienie głosów przed atakiem),
4. ``note_on`` (atak dźwięku),
5. ``note_off`` nut o zerowej długości (zamknięcie tuż po własnym ``note_on``).

Punkty 3-5 odpowiadają standardowi General MIDI (zwalniaj głosy przed atakiem),
z wyjątkiem nut zerowej długości, których ``note_off`` musi nastąpić po własnym
``note_on``. Parowanie ``note_on``/``note_off`` zakłada porządek FIFO względem
pary ``(channel, pitch)``. Ponieważ parser ponownie sortuje nuty wg
:func:`~musicians_style.midi.types.event_key`, dokładna kolejność ataków o tym
samym ``tick`` nie wpływa na wynikową *Reprezentację_Wewnętrzną*.

Rozkład zdarzeń na ścieżki
--------------------------
* **Format 0**: pojedyncza ścieżka zawierająca wszystkie zdarzenia.
* **Format 1**: ścieżka 0 (*conductor track*) z meta-zdarzeniami tempa, metrum
  i tonacji; ścieżka 1 ze zdarzeniami ``program_change`` oraz nutami.

W obu przypadkach znaczniki czasu są bezwzględne wewnątrz logiki printera i
dopiero na końcu konwertowane na czasy różnicowe (delta-time) wymagane przez
``mido``, dzięki czemu bezwzględne pozycje zdarzeń są zachowane (a więc i
round-trip parsera).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from mido import MidiFile, MidiTrack, Message, MetaMessage

from .types import InternalRepr, MetaEvent, NoteEvent

__all__ = ["MidiPrettyPrinter"]

# Domyślne tempo SMF: 500000 µs/ćwierćnutę == 120 BPM (Wymaganie 2.2).
_DEFAULT_TEMPO_US = 500_000

# Priorytety fazowe zdarzeń o identycznym znaczniku czasu (mniejszy = wcześniej).
_PHASE_META = 0
_PHASE_PROGRAM_CHANGE = 1
_PHASE_NOTE_OFF_SOUNDING = 2  # note_off nut o dodatniej długości
_PHASE_NOTE_ON = 3
_PHASE_NOTE_OFF_ZERO = 4  # note_off nut o zerowej długości

# Deterministyczna kolejność meta-zdarzeń o tym samym ticku.
_META_KIND_ORDER = {
    "tempo": 0,
    "time_signature": 1,
    "key_signature": 2,
}


class MidiPrettyPrinter:
    """Serializuje *Reprezentację_Wewnętrzną* do pliku Standard MIDI File.

    Klasa jest bezstanowa - można współdzielić jedną instancję. Wszystkie
    metody są czyste względem przekazanego :class:`InternalRepr` (nie modyfikują
    wejścia).
    """

    def write(self, repr_: InternalRepr, path: Path | str) -> None:
        """Zapisuje *Reprezentację_Wewnętrzną* do pliku MIDI pod ``path``.

        Args:
            repr_: *Reprezentacja_Wewnętrzna* do zserializowania.
            path: docelowa ścieżka pliku ``.mid``. Katalogi pośrednie są
                tworzone automatycznie.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        midi_file = self._build_midifile(repr_)
        midi_file.save(str(target))

    def to_bytes(self, repr_: InternalRepr) -> bytes:
        """Serializuje *Reprezentację_Wewnętrzną* do bajtów pliku MIDI.

        Args:
            repr_: *Reprezentacja_Wewnętrzna* do zserializowania.

        Returns:
            Zawartość pliku SMF jako ``bytes`` (rozpoczyna się od nagłówka
            ``b"MThd"``).
        """
        midi_file = self._build_midifile(repr_)
        buffer = io.BytesIO()
        midi_file.save(file=buffer)
        return buffer.getvalue()

    # -- budowa MidiFile -----------------------------------------------------

    def _build_midifile(self, repr_: InternalRepr) -> MidiFile:
        """Buduje obiekt ``mido.MidiFile`` z *Reprezentacji_Wewnętrznej*."""
        smf_format = repr_.smf_format
        if smf_format not in (0, 1):
            raise ValueError(
                "Nieobsługiwany format SMF: "
                f"{smf_format!r} (dozwolone wartości to 0 lub 1)."
            )
        if repr_.ticks_per_beat <= 0:
            raise ValueError(
                "ticks_per_beat musi być dodatnie, otrzymano: "
                f"{repr_.ticks_per_beat!r}."
            )

        midi_file = MidiFile(type=smf_format, ticks_per_beat=repr_.ticks_per_beat)

        tempo_meta = [m for m in repr_.meta if m.kind == "tempo"]
        timesig_meta = [m for m in repr_.meta if m.kind == "time_signature"]
        keysig_meta = [m for m in repr_.meta if m.kind == "key_signature"]
        program_meta = [m for m in repr_.meta if m.kind == "program_change"]
        global_meta = tempo_meta + timesig_meta + keysig_meta

        if smf_format == 0:
            # Wszystkie zdarzenia trafiają na jedną ścieżkę.
            events = self._collect_events(
                meta_events=global_meta,
                program_events=program_meta,
                notes=repr_.notes,
            )
            midi_file.tracks.append(self._build_track(events))
        else:
            # Format 1: conductor track (meta) + ścieżka nut.
            conductor_events = self._collect_events(
                meta_events=global_meta,
                program_events=[],
                notes=(),
            )
            note_events = self._collect_events(
                meta_events=[],
                program_events=program_meta,
                notes=repr_.notes,
            )
            midi_file.tracks.append(self._build_track(conductor_events))
            midi_file.tracks.append(self._build_track(note_events))

        return midi_file

    def _collect_events(
        self,
        *,
        meta_events: list[MetaEvent],
        program_events: list[MetaEvent],
        notes: tuple[NoteEvent, ...],
    ) -> list[tuple[tuple[int, ...], Any]]:
        """Tworzy listę par ``(sort_key, mido_message)`` o bezwzględnych tickach.

        ``sort_key`` ma postać ``(abs_tick, phase, sub_a, sub_b, sub_c, seq)`` i
        gwarantuje deterministyczny, totalny porządek (``seq`` to licznik
        zapewniający stabilność i eliminujący porównywanie obiektów wiadomości).
        """
        collected: list[tuple[tuple[int, ...], Any]] = []
        seq = 0

        for meta in meta_events:
            message = self._meta_to_message(meta)
            kind_order = _META_KIND_ORDER.get(meta.kind, len(_META_KIND_ORDER))
            key = (meta.tick, _PHASE_META, kind_order, 0, 0, seq)
            collected.append((key, message))
            seq += 1

        for meta in program_events:
            program = int(meta.payload.get("program", 0))
            channel = int(meta.payload.get("channel", 0))
            message = Message(
                "program_change", channel=channel, program=program, time=0
            )
            key = (meta.tick, _PHASE_PROGRAM_CHANGE, channel, program, 0, seq)
            collected.append((key, message))
            seq += 1

        for note in notes:
            on_message = Message(
                "note_on",
                channel=note.channel,
                note=note.pitch,
                velocity=note.velocity,
                time=0,
            )
            on_key = (
                note.tick,
                _PHASE_NOTE_ON,
                note.channel,
                note.pitch,
                note.velocity,
                seq,
            )
            collected.append((on_key, on_message))
            seq += 1

            off_tick = note.tick + note.duration_ticks
            off_phase = (
                _PHASE_NOTE_OFF_ZERO
                if note.duration_ticks == 0
                else _PHASE_NOTE_OFF_SOUNDING
            )
            off_message = Message(
                "note_off",
                channel=note.channel,
                note=note.pitch,
                velocity=0,
                time=0,
            )
            off_key = (off_tick, off_phase, note.channel, note.pitch, 0, seq)
            collected.append((off_key, off_message))
            seq += 1

        return collected

    @staticmethod
    def _meta_to_message(meta: MetaEvent) -> MetaMessage:
        """Konwertuje :class:`MetaEvent` na ``mido.MetaMessage`` (time=0)."""
        if meta.kind == "tempo":
            tempo_us = _resolve_tempo_us(meta.payload)
            return MetaMessage("set_tempo", tempo=tempo_us, time=0)
        if meta.kind == "time_signature":
            numerator = int(meta.payload.get("numerator", 4))
            denominator = int(meta.payload.get("denominator", 4))
            return MetaMessage(
                "time_signature",
                numerator=numerator,
                denominator=denominator,
                time=0,
            )
        if meta.kind == "key_signature":
            key = str(meta.payload.get("key", "C"))
            return MetaMessage("key_signature", key=key, time=0)
        raise ValueError(
            f"Nieobsługiwany rodzaj meta-zdarzenia na ścieżce conductor: {meta.kind!r}."
        )

    @staticmethod
    def _build_track(
        events: list[tuple[tuple[int, ...], Any]]
    ) -> MidiTrack:
        """Sortuje zdarzenia i konwertuje bezwzględne ticki na czasy różnicowe."""
        track = MidiTrack()
        events_sorted = sorted(events, key=lambda item: item[0])
        previous_tick = 0
        for key, message in events_sorted:
            abs_tick = key[0]
            message.time = abs_tick - previous_tick
            previous_tick = abs_tick
            track.append(message)
        return track


def _resolve_tempo_us(payload: dict[str, Any]) -> int:
    """Wyznacza tempo w mikrosekundach na ćwierćnutę z payloadu meta-zdarzenia.

    Preferowany jest klucz ``"tempo"`` (mikrosekundy). Dla zgodności wstecznej
    obsługiwany jest również klucz ``"bpm"`` (konwersja ``60_000_000 / bpm``).
    Brak obu kluczy skutkuje tempem domyślnym 120 BPM (``500000`` µs).
    """
    if "tempo" in payload:
        tempo_us = int(payload["tempo"])
    elif "bpm" in payload:
        bpm = float(payload["bpm"])
        if bpm <= 0:
            raise ValueError(f"Tempo (bpm) musi być dodatnie, otrzymano: {bpm!r}.")
        tempo_us = int(round(60_000_000 / bpm))
    else:
        tempo_us = _DEFAULT_TEMPO_US
    # SMF koduje tempo jako 24-bitową liczbę całkowitą (1..16_777_215).
    if not 1 <= tempo_us <= 0xFFFFFF:
        raise ValueError(
            f"Tempo poza zakresem SMF (1..16777215 µs/ćwierćnutę): {tempo_us!r}."
        )
    return tempo_us
