"""Dataset pianoroll i loader dla *Pipeline_Treningu* GAN (Wymaganie 3.2).

Moduł dostarcza dwa komponenty potoku treningowego *Modelu_GAN*:

* :class:`MultiArtistManifest` - agregacja *Manifestów_Zbioru* wielu artystów
  (*Zbiór_Wielo_Artystyczny*) wraz z deterministycznym mapowaniem
  ``artysta → indeks Etykiety_Artysty``. Wykorzystywana w trybie warunkowanym
  (StarGAN), w którym jeden generator ``G(x, c)`` obsługuje N artystów.
* :class:`PianorollDataset` - implementacja ``torch.utils.data.Dataset``
  przyjmująca pojedynczy :class:`~musicians_style.data.manifest.Manifest`
  (tryb per-artysta) lub :class:`MultiArtistManifest` (tryb warunkowany) i
  zwracająca pary ``(pianoroll, Etykieta_Artysty)`` gotowe do podania do
  :class:`~musicians_style.models.stargan.StarGANGenerator`.

Reprezentacja zwracanej próbki
------------------------------
``__getitem__`` zwraca krotkę ``(pianoroll, label)``:

* ``pianoroll`` - ``torch.FloatTensor`` o kształcie ``[1, T, P]`` (z wymiarem
  kanału), gdzie ``T`` to długość okna czasowego, a ``P`` liczba wysokości
  (domyślnie ``[1, 64, 84]``). Wymiar kanału jest wygodny dla generatora
  StarGAN, który oczekuje wejścia ``[B, 1, T, P]``.
* ``label`` - ``torch.FloatTensor`` *one-hot* o długości ``N`` (liczba
  artystów), reprezentujący *Etykietę_Artysty* ``c`` używaną do warunkowania.
  W trybie per-artysta (pojedynczy :class:`Manifest`) ``N == 1`` i etykieta to
  wektor ``[1.0]`` (indeks 0).

Rozwiązywanie ścieżek plików MIDI
---------------------------------
Wpisy :class:`~musicians_style.data.manifest.FileEntry` przechowują ścieżki
**względne** (względem katalogu *Zbioru_Stylu*). :class:`PianorollDataset`
przyjmuje więc katalog (lub mapowanie ``artysta → katalog``) służący do
rozwinięcia ścieżek względnych na bezwzględne. Ścieżki już bezwzględne są
wykorzystywane bez zmian, dzięki czemu dataset działa również z manifestami
przechowującymi pełne ścieżki.

Determinizm i leniwe parsowanie
-------------------------------
Kolejność próbek jest deterministyczna: artyści są uporządkowani według indeksu
*Etykiety_Artysty* (alfabetycznie po ``artist_id``), a w obrębie artysty pliki
zachowują kolejność z manifestu. Parsowanie plików MIDI (kosztowne wejście/
wyjście) jest **leniwe** - wykonywane dopiero w :meth:`PianorollDataset.__getitem__`,
opcjonalnie z prostym buforowaniem wyników (``cache=True``). Konstruktor nie
wykonuje ciężkich operacji wejścia/wyjścia.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.manifest import FileEntry, Manifest
from ..midi.parser import MidiParser
from ..midi.pianoroll import Pianoroll

__all__ = ["MultiArtistManifest", "PianorollDataset"]


class MultiArtistManifest:
    """Agregacja *Manifestów_Zbioru* wielu artystów (*Zbiór_Wielo_Artystyczny*).

    Klasa łączy manifesty per-artysta i nadaje każdemu artyście deterministyczny
    indeks *Etykiety_Artysty* (kolejność alfabetyczna ``artist_id``). Mapowanie
    ``indeks → artist_id`` (atrybut :attr:`artists`) odpowiada kolejności
    one-hot zapisywanej w metadanych *Punktu_Kontrolnego* (Wymaganie 3.14),
    dzięki czemu kolumna ``i`` *Etykiety_Artysty* zawsze odnosi się do tego
    samego artysty.

    Args:
        manifests: mapowanie ``artist_id → Manifest`` lub iterowalna kolekcja
            :class:`Manifest`. W obu przypadkach kolejność wejściowa nie ma
            znaczenia - indeksy są nadawane po posortowaniu ``artist_id``.

    Raises:
        ValueError: gdy kolekcja manifestów jest pusta albo gdy ``artist_id``
            powtarza się w kolekcji (niejednoznaczne mapowanie etykiet).

    Attributes:
        manifests: słownik ``artist_id → Manifest`` w kolejności indeksów etykiet.
        artists: lista ``artist_id`` uporządkowana wg indeksu *Etykiety_Artysty*.
    """

    def __init__(
        self, manifests: Mapping[str, Manifest] | Iterable[Manifest]
    ) -> None:
        collected: dict[str, Manifest] = {}

        if isinstance(manifests, Mapping):
            items: Iterable[tuple[str, Manifest]] = manifests.items()
            for artist_id, manifest in items:
                key = str(artist_id)
                if key in collected:
                    raise ValueError(
                        f"Zduplikowany artist_id w Zbiorze_Wielo_Artystycznym: {key!r}."
                    )
                collected[key] = manifest
        else:
            for manifest in manifests:
                key = str(manifest.artist_id)
                if key in collected:
                    raise ValueError(
                        f"Zduplikowany artist_id w Zbiorze_Wielo_Artystycznym: {key!r}."
                    )
                collected[key] = manifest

        if not collected:
            raise ValueError(
                "MultiArtistManifest wymaga co najmniej jednego manifestu artysty."
            )

        # Deterministyczne mapowanie artysta → indeks Etykiety_Artysty.
        self.artists: list[str] = sorted(collected)
        self.manifests: dict[str, Manifest] = {
            artist_id: collected[artist_id] for artist_id in self.artists
        }
        self._label_index: dict[str, int] = {
            artist_id: idx for idx, artist_id in enumerate(self.artists)
        }

    # -- API publiczne -------------------------------------------------------

    @classmethod
    def from_manifests(
        cls, manifests: Mapping[str, Manifest] | Iterable[Manifest]
    ) -> "MultiArtistManifest":
        """Tworzy :class:`MultiArtistManifest` z mapowania lub kolekcji manifestów."""
        return cls(manifests)

    @property
    def num_artists(self) -> int:
        """Liczba artystów ``N`` (wymiar one-hot *Etykiety_Artysty*)."""
        return len(self.artists)

    @property
    def total_files(self) -> int:
        """Łączna liczba wpisów plików we wszystkich manifestach."""
        return sum(len(manifest.files) for manifest in self.manifests.values())

    def label_index(self, artist_id: str) -> int:
        """Zwraca indeks *Etykiety_Artysty* dla danego ``artist_id``.

        Args:
            artist_id: identyfikator artysty.

        Returns:
            Indeks (0-based) odpowiadający kolumnie one-hot.

        Raises:
            KeyError: gdy ``artist_id`` nie należy do *Zbioru_Wielo_Artystycznego*.
        """
        return self._label_index[str(artist_id)]

    def __len__(self) -> int:
        """Liczba artystów (nie plików) - rozmiar mapowania etykiet."""
        return len(self.artists)

    def __repr__(self) -> str:  # pragma: no cover - reprezentacja pomocnicza
        return (
            f"MultiArtistManifest(artists={self.artists!r}, "
            f"total_files={self.total_files})"
        )


class PianorollDataset(Dataset):
    """Dataset zwracający pary ``(pianoroll, Etykieta_Artysty)`` (Wymaganie 3.2).

    Przyjmuje pojedynczy :class:`~musicians_style.data.manifest.Manifest`
    (tryb per-artysta, ``N == 1``) lub :class:`MultiArtistManifest` (tryb
    warunkowany, ``N`` artystów). Każdy element to krotka:

    * pianoroll ``torch.FloatTensor`` o kształcie ``[1, T, P]`` uzyskany z
      :meth:`~musicians_style.midi.pianoroll.Pianoroll.from_internal`,
    * *Etykieta_Artysty* one-hot ``torch.FloatTensor`` o długości ``N``.

    Parsowanie plików MIDI jest leniwe (w :meth:`__getitem__`) i może być
    buforowane (``cache=True``). Konstruktor jedynie buduje płaski, deterministyczny
    indeks ``(artist_id, FileEntry)`` bez czytania plików.

    Args:
        source: :class:`Manifest` lub :class:`MultiArtistManifest` opisujący
            *Zbiór_Stylu* / *Zbiór_Wielo_Artystyczny*.
        roots: katalog bazowy do rozwijania ścieżek względnych
            (:class:`~pathlib.Path` lub ``str``) albo mapowanie
            ``artist_id → katalog`` (dla wielu artystów o różnych katalogach).
            ``None`` oznacza brak prefiksu - ścieżki wpisów używane są bez zmian
            (wymaga wówczas ścieżek bezwzględnych lub względnych do bieżącego
            katalogu roboczego).
        pianoroll: konwerter :class:`~musicians_style.midi.pianoroll.Pianoroll`
            ustalający geometrię macierzy (``pitch_range``, ``steps_per_beat``,
            ``window_steps``). ``None`` → domyślny ``Pianoroll()`` (``[1, 64, 84]``).
        parser: :class:`~musicians_style.midi.parser.MidiParser` używany do
            wczytywania plików; ``None`` → domyślna instancja.
        cache: gdy ``True``, sparsowane pianorolle są buforowane w pamięci i
            ponownie wykorzystywane przy kolejnych odczytach tego samego indeksu.

    Raises:
        TypeError: gdy ``source`` nie jest :class:`Manifest` ani
            :class:`MultiArtistManifest`.
    """

    def __init__(
        self,
        source: Manifest | MultiArtistManifest,
        roots: Path | str | Mapping[str, Path | str] | None = None,
        pianoroll: Pianoroll | None = None,
        parser: MidiParser | None = None,
        cache: bool = False,
    ) -> None:
        if isinstance(source, MultiArtistManifest):
            self._multi = source
        elif isinstance(source, Manifest):
            # Tryb per-artysta sprowadzamy do jednoelementowego
            # Zbioru_Wielo_Artystycznego (N == 1, indeks etykiety 0).
            self._multi = MultiArtistManifest([source])
        else:
            raise TypeError(
                "source musi być Manifest lub MultiArtistManifest, otrzymano: "
                f"{type(source).__name__}."
            )

        self._pianoroll = pianoroll if pianoroll is not None else Pianoroll()
        self._parser = parser if parser is not None else MidiParser()
        self._roots = self._normalize_roots(roots)
        self._cache_enabled = bool(cache)
        self._cache: dict[int, torch.Tensor] = {}

        # Płaski, deterministyczny indeks: (artist_id, FileEntry).
        # Artyści wg indeksu Etykiety_Artysty, pliki w kolejności z manifestu.
        self._index: list[tuple[str, FileEntry]] = []
        for artist_id in self._multi.artists:
            manifest = self._multi.manifests[artist_id]
            for entry in manifest.files:
                self._index.append((artist_id, entry))

    # -- właściwości pomocnicze ---------------------------------------------

    @property
    def artists(self) -> list[str]:
        """Lista ``artist_id`` uporządkowana wg indeksu *Etykiety_Artysty*."""
        return list(self._multi.artists)

    @property
    def num_artists(self) -> int:
        """Liczba artystów ``N`` (długość one-hot *Etykiety_Artysty*)."""
        return self._multi.num_artists

    @property
    def manifest(self) -> MultiArtistManifest:
        """Agregowany *Zbiór_Wielo_Artystyczny* leżący u podstaw datasetu."""
        return self._multi

    # -- protokół Dataset ----------------------------------------------------

    def __len__(self) -> int:
        """Łączna liczba próbek (plików) we wszystkich manifestach."""
        return len(self._index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Zwraca parę ``(pianoroll [1, T, P], Etykieta_Artysty one-hot [N])``.

        Args:
            index: indeks próbki w zakresie ``[0, len(self))``. Akceptowane są
                indeksy ujemne (semantyka jak w listach Pythona).

        Returns:
            Krotka ``(pianoroll, label)``:

            * ``pianoroll`` - ``torch.FloatTensor`` ``[1, T, P]``,
            * ``label`` - ``torch.FloatTensor`` one-hot o długości ``N``.

        Raises:
            IndexError: gdy ``index`` jest poza zakresem.
        """
        if index < 0:
            index += len(self._index)
        if not (0 <= index < len(self._index)):
            raise IndexError(
                f"indeks {index} poza zakresem datasetu (rozmiar {len(self._index)})."
            )

        artist_id, entry = self._index[index]
        pianoroll_tensor = self._load_pianoroll(index, artist_id, entry)
        label = self._one_hot(self._multi.label_index(artist_id))
        return pianoroll_tensor, label

    # -- ładowanie i konwersja ----------------------------------------------

    def _load_pianoroll(
        self, index: int, artist_id: str, entry: FileEntry
    ) -> torch.Tensor:
        """Parsuje plik MIDI i zwraca pianoroll ``[1, T, P]`` (z buforowaniem)."""
        if self._cache_enabled and index in self._cache:
            return self._cache[index]

        path = self._resolve_path(artist_id, entry.path)
        repr_ = self._parser.parse(path)
        roll = self._pianoroll.from_internal(repr_)  # [T, P] float32 numpy
        tensor = torch.from_numpy(np.ascontiguousarray(roll, dtype=np.float32))
        tensor = tensor.unsqueeze(0)  # [1, T, P] - wymiar kanału dla StarGAN

        if self._cache_enabled:
            self._cache[index] = tensor
        return tensor

    def _one_hot(self, label_index: int) -> torch.Tensor:
        """Buduje wektor one-hot *Etykiety_Artysty* o długości ``N``."""
        label = torch.zeros(self._multi.num_artists, dtype=torch.float32)
        label[label_index] = 1.0
        return label

    # -- rozwiązywanie ścieżek ----------------------------------------------

    @staticmethod
    def _normalize_roots(
        roots: Path | str | Mapping[str, Path | str] | None,
    ) -> dict[str, Path] | Path | None:
        """Normalizuje argument ``roots`` do ``None`` / ``Path`` / ``dict``."""
        if roots is None:
            return None
        if isinstance(roots, Mapping):
            return {str(k): Path(v) for k, v in roots.items()}
        return Path(roots)

    def _resolve_path(self, artist_id: str, rel_path: str) -> Path:
        """Rozwija ścieżkę wpisu do ścieżki bezwzględnej zgodnie z ``roots``.

        Ścieżki już bezwzględne są zwracane bez zmian. Dla ścieżek względnych
        stosowany jest katalog bazowy: wspólny (``Path``) albo per-artysta
        (``dict``); brak ``roots`` oznacza użycie ścieżki bez prefiksu.
        """
        candidate = Path(rel_path)
        if candidate.is_absolute():
            return candidate

        if self._roots is None:
            return candidate
        if isinstance(self._roots, dict):
            base = self._roots.get(artist_id)
            if base is None:
                return candidate
            return base / candidate
        return self._roots / candidate
