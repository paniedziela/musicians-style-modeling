"""Hierarchia wyjątków *Systemu* modelowania stylu artystycznego muzyków.

Moduł definiuje bazowy wyjątek :class:`MusiciansStyleError` oraz wyspecjalizowane
podtypy odpowiadające trzem warstwom błędów opisanym w sekcji *Error Handling*
dokumentu ``design.md``:

1. **Błędy walidacji wejścia** - :class:`MidiValidationError`,
   :class:`EmptyDatasetError`, :class:`ConfigValidationError`.
2. **Błędy zasobów** - :class:`GpuOutOfMemoryError`.
3. **Błędy logiki domeny** - :class:`UnknownArtistError`,
   :class:`ArtistMismatchError`.

Każdy wyjątek niesie pole ``exit_code`` zgodne z sekcją *Kody wyjścia CLI*
(``2`` - błąd walidacji wejścia/konfiguracji lub logiki domeny, ``3`` - błąd
zasobów). Warstwa CLI może użyć tej wartości do zakończenia procesu właściwym
kodem (Wymagania 1.5, 3.9, 5.9, 5.10).
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "MIDI_VALIDATION_DEFAULT_MESSAGE",
    "MusiciansStyleError",
    "MidiValidationError",
    "EmptyDatasetError",
    "UnknownArtistError",
    "ArtistMismatchError",
    "GpuOutOfMemoryError",
    "ConfigValidationError",
]

#: Niepusty, ogólny opis błędu walidacji MIDI używany, gdy charakter uszkodzenia
#: uniemożliwia precyzyjne wskazanie miejsca lub typu (Wymaganie 5.5).
MIDI_VALIDATION_DEFAULT_MESSAGE = "wykryto uszkodzenie struktury pliku"


class MusiciansStyleError(Exception):
    """Bazowy wyjątek *Systemu*.

    Wszystkie wyjątki domenowe dziedziczą po tej klasie, co pozwala warstwie CLI
    przechwycić je jednym blokiem ``except MusiciansStyleError`` i zakończyć
    działanie kodem wyjścia ``exit_code``.
    """

    #: Domyślny kod wyjścia CLI dla nieskategoryzowanego błędu domenowego.
    exit_code: int = 1


class MidiValidationError(MusiciansStyleError):
    """Błąd walidacji struktury pliku MIDI (Wymaganie 5.5).

    Pole ``message`` jest **zawsze niepuste**: jeżeli wywołujący nie poda opisu
    (pusty łańcuch lub ``None``), używana jest stała
    :data:`MIDI_VALIDATION_DEFAULT_MESSAGE`. Niezmiennik ten jest dodatkowo
    chroniony asercją, co gwarantuje własność poprawnościową *Property 13*
    (niepusty opis błędu walidacji MIDI).
    """

    exit_code: int = 2

    def __init__(
        self,
        message: str | None = None,
        file: Path | str | None = None,
        offset: int | None = None,
    ) -> None:
        # Wymaganie 5.5: każde wystąpienie błędu walidacji MUSI mieć niepusty opis.
        resolved = message if message else MIDI_VALIDATION_DEFAULT_MESSAGE
        assert resolved, "MidiValidationError MUST have non-empty message"

        self.message: str = resolved
        self.file: Path | None = Path(file) if file is not None else None
        self.offset: int | None = offset

        detail = resolved
        if self.file is not None:
            detail = f"{detail} (plik: {self.file})"
        if offset is not None:
            detail = f"{detail} (offset: {offset})"
        super().__init__(detail)


class EmptyDatasetError(MusiciansStyleError):
    """Brak poprawnych plików MIDI po walidacji zbioru (Wymaganie 1.5).

    Zgłaszany, gdy liczba poprawnych plików w *Zbiorze_Stylu* wynosi zero -
    niezależnie od przyczyny (pusty katalog, odrzucenie z powodu niezgodności ze
    SMF, przekroczenie dozwolonego zakresu długości itd.).
    """

    exit_code: int = 2


class UnknownArtistError(MusiciansStyleError):
    """Żądany ``target_artist`` spoza listy obsługiwanej przez *Punkt_Kontrolny*.

    Odpowiada Wymaganiu 5.9 (tryb ``conditional``). Niesie nazwę żądanego artysty
    oraz listę dostępnych *Etykiet_Artysty*, dzięki czemu komunikat błędu wskazuje
    użytkownikowi poprawne wartości.
    """

    exit_code: int = 2

    def __init__(self, requested: str, available: list[str]) -> None:
        self.requested = requested
        self.available = list(available)
        super().__init__(
            f"Nieznany artysta '{requested}'. Dostępni artyści: {self.available}"
        )


class ArtistMismatchError(MusiciansStyleError):
    """Niezgodność ``target_artist`` z *Punktem_Kontrolnym* trybu ``per_artist``.

    Odpowiada Wymaganiu 5.10: *Punkt_Kontrolny* zapisany dla pojedynczego artysty
    został użyty z innym artystą docelowym.
    """

    exit_code: int = 2

    def __init__(self, requested: str, expected: str) -> None:
        self.requested = requested
        self.expected = expected
        super().__init__(
            f"Punkt_Kontrolny trybu per_artist obsługuje artystę '{expected}', "
            f"żądano '{requested}'"
        )


class GpuOutOfMemoryError(MusiciansStyleError):
    """Niewystarczająca pamięć GPU w trakcie treningu (Wymaganie 3.9).

    Niesie sugerowany zredukowany rozmiar wsadu (``suggested_batch_size``), który
    *Pipeline_Treningu* może zaproponować użytkownikowi przed zakończeniem
    działania kodem zasobowym ``3``.
    """

    exit_code: int = 3

    def __init__(
        self,
        message: str = "Niewystarczająca pamięć GPU (CUDA out of memory)",
        suggested_batch_size: int | None = None,
    ) -> None:
        self.suggested_batch_size = suggested_batch_size
        if suggested_batch_size is not None:
            message = f"{message}; sugerowana redukcja batch_size do {suggested_batch_size}"
        super().__init__(message)


class ConfigValidationError(MusiciansStyleError):
    """Niespójność lub brak wymaganych pól w pliku konfiguracyjnym.

    Zgłaszany przez loader konfiguracji (``config.py``) m.in. przy braku pól
    ``seed``, ``model.mode`` lub ``model.artists`` dla trybu ``conditional``.
    """

    exit_code: int = 2
