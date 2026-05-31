"""Parser_MIDI - konwersja pliku Standard MIDI File na *Reprezentację_Wewnętrzną*.

Moduł implementuje :class:`MidiParser` - komponent odwrotny do
:class:`~musicians_style.midi.printer.MidiPrettyPrinter`. Wczytuje plik
*Standard MIDI File* (SMF, format 0 lub 1) i przekształca go na
:class:`~musicians_style.midi.types.InternalRepr` z deterministycznie
posortowanymi krotkami zdarzeń (Wymagania 5.3, 5.5).

Walidacja struktury przed przetwarzaniem treści (Wymaganie 5.5)
--------------------------------------------------------------
Metoda :meth:`MidiParser.validate` operuje **wyłącznie na surowych bajtach**
i jest wykonywana zanim nastąpi jakiekolwiek przetwarzanie treści (w
szczególności przed obliczaniem długości czasowej). Kontroluje:

* obecność i poprawność nagłówka ``MThd`` (sygnatura, długość >= 6),
* zadeklarowany format SMF (0, 1 lub 2) i liczbę ścieżek ``ntrks``,
* strukturę chunków ``MTrk`` (sygnatura, długość bloku mieszcząca się w pliku),
* zgodność liczby napotkanych ścieżek z nagłówkiem.

Każde wykryte uszkodzenie skutkuje zgłoszeniem
:class:`~musicians_style.errors.MidiValidationError` z **niepustym** opisem
miejsca i typu uszkodzenia; gdy lokalizacja jest niemożliwa, używana jest
stała :data:`~musicians_style.errors.MIDI_VALIDATION_DEFAULT_MESSAGE`
(„wykryto uszkodzenie struktury pliku"). Walidacja nigdy nie zgłasza
nieobsłużonego wyjątku dla dowolnych danych binarnych (Property 13).

Schemat payloadów meta-zdarzeń (kontrakt z Pretty_Printer_MIDI)
---------------------------------------------------------------
Parser produkuje payloady zgodne ze schematem oczekiwanym przez
:class:`MidiPrettyPrinter`, dzięki czemu zachowana jest własność round-trip
(Property 1, Wymaganie 5.4):

* ``tempo``           → ``{"tempo": <mikrosekundy na ćwierćnutę>}``.
* ``time_signature``  → ``{"numerator": <int>, "denominator": <int>}``.
* ``key_signature``   → ``{"key": "<nazwa tonacji mido>"}``.
* ``program_change``  → ``{"program": <0..127>, "channel": <0..15>}``.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from mido import MidiFile

from ..errors import MidiValidationError
from .types import InternalRepr, MetaEvent, NoteEvent, event_key

__all__ = ["MidiParser"]

# Minimalny rozmiar nagłówka MThd: 4 bajty sygnatury + 4 bajty długości + 6 bajtów treści.
_MTHD_MIN_TOTAL = 14
# Minimalna zadeklarowana długość treści nagłówka MThd (format + ntrks + division).
_MTHD_MIN_BODY = 6
# Rozmiar nagłówka pojedynczego chunku: sygnatura (4) + długość (4).
_CHUNK_HEADER = 8
# Dozwolone formaty SMF.
_VALID_SMF_FORMATS = (0, 1, 2)


class MidiParser:
    """Wczytuje pliki MIDI i konwertuje je na *Reprezentację_Wewnętrzną*.

    Klasa jest bezstanowa - jedną instancję można bezpiecznie współdzielić.
    Wszystkie metody są czyste względem wejścia (nie mutują argumentów).
    """

    # -- API publiczne -------------------------------------------------------

    def parse(self, path: Path | str) -> InternalRepr:
        """Wczytuje plik MIDI spod ``path`` i zwraca *Reprezentację_Wewnętrzną*.

        Args:
            path: ścieżka do pliku ``.mid`` / ``.midi``.

        Returns:
            :class:`InternalRepr` z deterministycznie posortowanymi krotkami
            zdarzeń nutowych i meta.

        Raises:
            MidiValidationError: gdy struktura pliku jest uszkodzona
                (Wymaganie 5.5).
        """
        data = self._read_bytes(path)
        return self._parse_validated(data, source=Path(path))

    def parse_bytes(self, data: bytes) -> InternalRepr:
        """Parsuje surowe bajty pliku MIDI i zwraca *Reprezentację_Wewnętrzną*.

        Args:
            data: zawartość pliku SMF jako ``bytes``.

        Returns:
            :class:`InternalRepr` z deterministycznie posortowanymi krotkami.

        Raises:
            MidiValidationError: gdy struktura danych jest uszkodzona.
        """
        return self._parse_validated(bytes(data), source=None)

    def validate(self, source: Path | str | bytes) -> None:
        """Waliduje strukturę pliku MIDI na podstawie surowych bajtów.

        Walidacja jest wykonywana przed jakimkolwiek przetwarzaniem treści
        (Wymaganie 5.5). Przyjmuje zarówno ścieżkę pliku, jak i bezpośrednio
        bajty (wykorzystywane m.in. w teście własnościowym Property 13).

        Args:
            source: ścieżka do pliku (``Path``/``str``) albo surowe ``bytes``.

        Raises:
            MidiValidationError: z niepustym opisem, gdy wykryto uszkodzenie
                struktury (nagłówek, długości bloków, struktura chunków).
        """
        if isinstance(source, (bytes, bytearray)):
            self._validate_bytes(bytes(source))
            return
        data = self._read_bytes(source)
        self._validate_bytes(data)

    # -- walidacja struktury (surowe bajty) ----------------------------------

    def _validate_bytes(self, data: bytes) -> None:
        """Sprawdza strukturę SMF na poziomie chunków, bez przetwarzania treści.

        Implementacja jest defensywna - dla dowolnych danych binarnych zgłasza
        wyłącznie :class:`MidiValidationError` (nigdy ``IndexError`` /
        ``TypeError`` / ``AttributeError``), co realizuje Property 13.
        """
        # --- nagłówek MThd ---
        if len(data) < _MTHD_MIN_TOTAL:
            raise MidiValidationError(
                "plik krótszy niż nagłówek MThd "
                f"(wymagane min. {_MTHD_MIN_TOTAL} bajtów, jest {len(data)})",
                offset=0,
            )
        if data[0:4] != b"MThd":
            raise MidiValidationError(
                "brak sygnatury nagłówka 'MThd' na początku pliku",
                offset=0,
            )

        header_len = int.from_bytes(data[4:8], "big")
        if header_len < _MTHD_MIN_BODY:
            raise MidiValidationError(
                f"zadeklarowana długość nagłówka MThd ({header_len}) jest mniejsza "
                f"niż wymagane {_MTHD_MIN_BODY} bajtów",
                offset=4,
            )

        smf_format = int.from_bytes(data[8:10], "big")
        ntrks = int.from_bytes(data[10:12], "big")
        if smf_format not in _VALID_SMF_FORMATS:
            raise MidiValidationError(
                f"nieobsługiwany format SMF: {smf_format} (dozwolone 0, 1, 2)",
                offset=8,
            )
        if smf_format == 0 and ntrks != 1:
            raise MidiValidationError(
                f"format 0 wymaga dokładnie jednej ścieżki, zadeklarowano {ntrks}",
                offset=10,
            )

        # Treść nagłówka może być dłuższa niż 6 bajtów - pomijamy zadeklarowaną długość.
        pos = _CHUNK_HEADER + header_len
        if pos > len(data) and ntrks > 0:
            raise MidiValidationError(
                "zadeklarowana długość nagłówka MThd wykracza poza rozmiar pliku, "
                "brak miejsca na ścieżki",
                offset=4,
            )

        # --- chunki ścieżek MTrk ---
        track_count = 0
        while pos < len(data):
            if pos + _CHUNK_HEADER > len(data):
                raise MidiValidationError(
                    "niekompletny nagłówek chunku ścieżki "
                    f"(oczekiwano {_CHUNK_HEADER} bajtów sygnatury i długości)",
                    offset=pos,
                )
            chunk_id = data[pos : pos + 4]
            chunk_len = int.from_bytes(data[pos + 4 : pos + 8], "big")
            if chunk_id != b"MTrk":
                raise MidiValidationError(
                    f"nieoczekiwana sygnatura chunku {chunk_id!r}, oczekiwano 'MTrk'",
                    offset=pos,
                )
            data_end = pos + _CHUNK_HEADER + chunk_len
            if data_end > len(data):
                raise MidiValidationError(
                    f"zadeklarowana długość ścieżki ({chunk_len}) wykracza poza "
                    f"rozmiar pliku (offset {pos})",
                    offset=pos + 4,
                )
            track_count += 1
            pos = data_end

        if track_count != ntrks:
            raise MidiValidationError(
                f"liczba napotkanych ścieżek ({track_count}) niezgodna z nagłówkiem "
                f"MThd (zadeklarowano {ntrks})",
                offset=10,
            )

    # -- parsowanie treści ---------------------------------------------------

    def _parse_validated(self, data: bytes, source: Path | None) -> InternalRepr:
        """Waliduje strukturę, a następnie konwertuje treść na ``InternalRepr``."""
        # Wymaganie 5.5: walidacja struktury przed przetwarzaniem treści.
        self._validate_bytes(data)

        try:
            midi_file = MidiFile(file=io.BytesIO(data))
        except MidiValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - mido sygnalizuje błędy różnymi typami
            # Plik przeszedł walidację struktury chunków, lecz treść jest błędna
            # (np. niepoprawny komunikat, uszkodzony running status). Opakowujemy
            # w MidiValidationError z niepustym opisem (Wymaganie 5.5).
            raise MidiValidationError(
                f"błąd podczas dekodowania treści MIDI: {exc}",
                file=source,
            ) from exc

        return self._build_internal_repr(midi_file)

    def _build_internal_repr(self, midi_file: MidiFile) -> InternalRepr:
        """Buduje ``InternalRepr`` z obiektu ``mido.MidiFile``."""
        notes: list[NoteEvent] = []
        meta: list[MetaEvent] = []

        for track in midi_file.tracks:
            track_notes, track_meta = self._convert_track(track)
            notes.extend(track_notes)
            meta.extend(track_meta)

        sorted_notes = tuple(sorted(notes, key=event_key))
        sorted_meta = tuple(sorted(meta, key=_meta_key))

        smf_format = midi_file.type if midi_file.type in (0, 1) else 1
        ticks_per_beat = midi_file.ticks_per_beat or 480

        return InternalRepr(
            ticks_per_beat=ticks_per_beat,
            notes=sorted_notes,
            meta=sorted_meta,
            smf_format=smf_format,
        )

    def _convert_track(
        self, track: Any
    ) -> tuple[list[NoteEvent], list[MetaEvent]]:
        """Konwertuje pojedynczą ścieżkę mido na listy zdarzeń nutowych i meta.

        Czasy delta są akumulowane do czasów bezwzględnych. Pary
        ``note_on``/``note_off`` (dla danego ``(channel, pitch)``) są parowane w
        kolejności FIFO; ``note_on`` z ``velocity == 0`` jest traktowany jak
        ``note_off`` zgodnie ze specyfikacją MIDI 1.0.
        """
        notes: list[NoteEvent] = []
        meta: list[MetaEvent] = []
        # (channel, pitch) -> lista (start_tick, velocity) otwartych nut (FIFO).
        open_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}

        abs_tick = 0
        for message in track:
            abs_tick += int(getattr(message, "time", 0) or 0)

            msg_type = message.type
            if msg_type == "note_on" and message.velocity > 0:
                key = (message.channel, message.note)
                open_notes.setdefault(key, []).append((abs_tick, message.velocity))
            elif msg_type == "note_off" or (
                msg_type == "note_on" and message.velocity == 0
            ):
                key = (message.channel, message.note)
                started = open_notes.get(key)
                if started:
                    start_tick, velocity = started.pop(0)
                    notes.append(
                        NoteEvent(
                            tick=start_tick,
                            channel=message.channel,
                            pitch=message.note,
                            velocity=velocity,
                            duration_ticks=abs_tick - start_tick,
                        )
                    )
            elif msg_type == "program_change":
                meta.append(
                    MetaEvent(
                        tick=abs_tick,
                        kind="program_change",
                        payload={
                            "program": int(message.program),
                            "channel": int(message.channel),
                        },
                    )
                )
            elif msg_type == "set_tempo":
                meta.append(
                    MetaEvent(
                        tick=abs_tick,
                        kind="tempo",
                        payload={"tempo": int(message.tempo)},
                    )
                )
            elif msg_type == "time_signature":
                meta.append(
                    MetaEvent(
                        tick=abs_tick,
                        kind="time_signature",
                        payload={
                            "numerator": int(message.numerator),
                            "denominator": int(message.denominator),
                        },
                    )
                )
            elif msg_type == "key_signature":
                meta.append(
                    MetaEvent(
                        tick=abs_tick,
                        kind="key_signature",
                        payload={"key": str(message.key)},
                    )
                )

        # Nuty bez pasującego note_off domykamy na końcu ścieżki (bezwzględny tick).
        for (channel, pitch), started in open_notes.items():
            for start_tick, velocity in started:
                notes.append(
                    NoteEvent(
                        tick=start_tick,
                        channel=channel,
                        pitch=pitch,
                        velocity=velocity,
                        duration_ticks=max(0, abs_tick - start_tick),
                    )
                )

        return notes, meta

    # -- pomocnicze ----------------------------------------------------------

    @staticmethod
    def _read_bytes(path: Path | str) -> bytes:
        """Wczytuje zawartość pliku jako bajty (na potrzeby walidacji/parsowania)."""
        return Path(path).read_bytes()


def _meta_key(meta: MetaEvent) -> tuple[int, str, tuple[Any, ...]]:
    """Deterministyczny klucz sortowania zdarzeń meta.

    Sortuje po ``(tick, kind, posortowane elementy payloadu)``, co zapewnia
    powtarzalną, stabilną kolejność niezależnie od kolejności wczytania.
    """
    payload_items = tuple(sorted((str(k), repr(v)) for k, v in meta.payload.items()))
    return (meta.tick, meta.kind, payload_items)
