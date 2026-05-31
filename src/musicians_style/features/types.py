"""Model danych *Wektora_Cech* i agregatów statystycznych (Wymagania 2.1, 2.3).

Moduł definiuje niemutowalne struktury danych *Ekstraktora_Cech* (sekcja
*Components and Interfaces* w ``design.md``):

- :class:`FeatureVector` - *Wektor_Cech* pojedynczego pliku MIDI.
- :class:`AggregatedFeatures` - agregat statystyczny *Zbioru_Stylu* (średnia,
  mediana, odchylenie standardowe oraz macierz kowariancji do odległości
  Mahalanobisa, Wymaganie 4.3).

Kluczowe decyzje projektowe
===========================

**1. Niemutowalność i tablice NumPy.**
Obie struktury są ``@dataclass(frozen=True, eq=False)``. ``frozen=True`` blokuje
przypisywanie atrybutów (``__setattr__``/``__delattr__`` zgłaszają
``FrozenInstanceError``), co czyni instancje koncepcyjnie niemutowalnymi. Tablice
NumPy (histogramy, kowariancja) nie są jednak haszowalne ani porównywalne
operatorem ``==`` zwracającym pojedynczą wartość logiczną, dlatego:

- ``eq=False`` wyłącza automatyczne generowanie ``__eq__`` przez ``dataclass``;
  definiujemy własne ``__eq__`` porównujące pola skalarne dokładnie (``==``),
  a tablice przez :func:`numpy.array_equal` (z ``equal_nan=True``). Dzięki temu
  test idempotencji (*Property 2*, zadanie 3.3) może asertować bit-identyczną
  równość dwóch wywołań ``extract(r)``.
- przy ``eq=False`` ``dataclass`` pozostawia ``__hash__`` nietknięty, więc
  definiujemy własny ``__hash__`` spójny z ``__eq__`` (oparty na ``tobytes()``
  tablic), aby instancje pozostały użyteczne w zbiorach i słownikach.
- w ``__post_init__`` wszystkie tablice są kopiowane do tablic ``float64``,
  których właściwość ``writeable`` jest ustawiana na ``False``
  (``arr.flags.writeable = False``). Gwarantuje to, że dane liczbowe nie zostaną
  zmodyfikowane "w miejscu" po utworzeniu wektora, zachowując pełną
  niemutowalność niezależnie od tablicy źródłowej przekazanej do konstruktora.

**2. Walidacja w warstwie typu.**
``__post_init__`` egzekwuje wyłącznie **kształty** tablic (``(12,)`` dla
histogramu klas wysokości, ``(25,)`` dla histogramu interwałów). Niezmienniki
wartościowe (``tempo_bpm > 0``, ``rest_ratio ∈ [0, 1]``, suma histogramu klas
wysokości równa 1) są gwarantowane przez *Ekstraktor_Cech* (zadanie 3.2) i
weryfikowane testem *Property 3* (zadanie 3.4); nie są wymuszane tutaj, aby móc
reprezentować również wektory częściowe (Wymaganie 2.8).

**3. Reprezentacja wektorowa (** :meth:`FeatureVector.as_array` **).**
``as_array`` zwraca jednowymiarową tablicę ``float64`` o **stałej długości**
:data:`FEATURE_VECTOR_LENGTH` (= 42), niezależnej od pliku wejściowego
(Wymaganie 2.3, *Property 3*). Pole ``key`` (kategoryczne) jest **wyłączone** z
tej reprezentacji - patrz uzasadnienie przy :meth:`FeatureVector.as_array`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "PITCH_CLASS_BINS",
    "INTERVAL_HISTOGRAM_BINS",
    "FEATURE_VECTOR_LENGTH",
    "FeatureVector",
    "AggregatedFeatures",
]

#: Liczba klas wysokości dźwięku (chromatyczne C..B) - kształt
#: ``pitch_class_histogram`` to ``(PITCH_CLASS_BINS,)``.
PITCH_CLASS_BINS = 12

#: Liczba kubełków histogramu interwałów melodycznych: interwały od -12 do +12
#: półtonów włącznie (25 wartości). Semantyka poszczególnych kubełków jest
#: ustalana przez *Ekstraktor_Cech* (zadanie 3.2); tutaj egzekwowany jest jedynie
#: kształt ``(INTERVAL_HISTOGRAM_BINS,)``.
INTERVAL_HISTOGRAM_BINS = 25

#: Stała długość wektora zwracanego przez :meth:`FeatureVector.as_array`.
#: Układ (patrz docstring metody): tempo (1) + histogram klas wysokości (12) +
#: histogram interwałów (25) + gęstość nut (1) + średnia długość nuty (1) +
#: odchylenie długości nuty (1) + proporcja pauz (1) = 42. Pole ``key`` jest
#: wyłączone z reprezentacji wektorowej.
FEATURE_VECTOR_LENGTH = (
    1 + PITCH_CLASS_BINS + INTERVAL_HISTOGRAM_BINS + 1 + 1 + 1 + 1
)


def _as_readonly_float_array(
    value: Any, expected_shape: tuple[int, ...], name: str
) -> np.ndarray:
    """Zwraca kopię ``value`` jako tablicę ``float64`` tylko do odczytu.

    Tworzy nową tablicę (kopię, której dane należą do zwracanego obiektu),
    waliduje jej kształt względem ``expected_shape`` i ustawia
    ``flags.writeable = False``, dzięki czemu nie da się jej zmodyfikować w
    miejscu. Wymuszenie ``dtype=float64`` zapewnia deterministyczne porównania i
    haszowanie (jednolity typ i układ bajtów).

    Raises:
        ValueError: gdy kształt tablicy różni się od ``expected_shape``.
    """
    arr = np.array(value, dtype=np.float64)  # kopia: dane należą do `arr`
    if arr.shape != expected_shape:
        raise ValueError(
            f"Pole '{name}' musi mieć kształt {expected_shape}, "
            f"otrzymano {arr.shape}."
        )
    arr.flags.writeable = False
    return arr


def _arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Porównuje dwie tablice element-po-elemencie (``NaN`` == ``NaN``).

    Używane przez ``__eq__``: dwie tablice są równe, gdy mają identyczny kształt
    i identyczne wartości. ``equal_nan=True`` sprawia, że pozycje ``NaN`` w obu
    tablicach są traktowane jako równe - dzięki temu test idempotencji
    (*Property 2*) działa również wtedy, gdy *Ekstraktor_Cech* zwróci wartości
    ``NaN`` (np. wektor częściowy, Wymaganie 2.8).
    """
    if a.shape != b.shape:
        return False
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:
        # equal_nan nie jest wspierane dla typów nie-zmiennoprzecinkowych.
        return bool(np.array_equal(a, b))


@dataclass(frozen=True, eq=False)
class FeatureVector:
    """*Wektor_Cech* pojedynczego pliku MIDI (Wymaganie 2.1).

    Atrybuty:
        tempo_bpm: tempo średnie w BPM (niezmiennik *Ekstraktora_Cech*: > 0).
        key: tonacja dominująca jako etykieta tekstowa, np. ``"C major"``,
            ``"A minor"`` (cecha kategoryczna).
        pitch_class_histogram: histogram klas wysokości dźwięków, kształt
            ``(12,)``; niezmiennik *Ekstraktora_Cech*: suma == 1 (± 1e-9).
        interval_histogram: histogram interwałów melodycznych, kształt ``(25,)``
            (interwały -12..+12 półtonów; semantyka kubełków ustalana w
            *Ekstraktorze_Cech*).
        note_density_per_s: gęstość nut na sekundę (niezmiennik: >= 0).
        mean_note_duration_s: średnia długość nuty w sekundach (>= 0).
        std_note_duration_s: odchylenie standardowe długości nuty (>= 0).
        rest_ratio: proporcja pauz (niezmiennik: w przedziale [0, 1]).

    Instancje są niemutowalne (``frozen=True``); tablice są kopiowane do
    ``float64`` tylko do odczytu w ``__post_init__``. Równość i haszowanie są
    zdefiniowane ręcznie (patrz docstring modułu).
    """

    tempo_bpm: float
    key: str
    pitch_class_histogram: np.ndarray
    interval_histogram: np.ndarray
    note_density_per_s: float
    mean_note_duration_s: float
    std_note_duration_s: float
    rest_ratio: float

    def __post_init__(self) -> None:
        # Normalizacja pól skalarnych do typów wbudowanych (deterministyczne
        # porównania, brak np.float64 "przeciekającego" do równości/haszowania).
        object.__setattr__(self, "tempo_bpm", float(self.tempo_bpm))
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(self, "note_density_per_s", float(self.note_density_per_s))
        object.__setattr__(
            self, "mean_note_duration_s", float(self.mean_note_duration_s)
        )
        object.__setattr__(
            self, "std_note_duration_s", float(self.std_note_duration_s)
        )
        object.__setattr__(self, "rest_ratio", float(self.rest_ratio))

        # Kopie tablic tylko do odczytu o wymuszonym kształcie i dtype float64.
        object.__setattr__(
            self,
            "pitch_class_histogram",
            _as_readonly_float_array(
                self.pitch_class_histogram,
                (PITCH_CLASS_BINS,),
                "pitch_class_histogram",
            ),
        )
        object.__setattr__(
            self,
            "interval_histogram",
            _as_readonly_float_array(
                self.interval_histogram,
                (INTERVAL_HISTOGRAM_BINS,),
                "interval_histogram",
            ),
        )

    def as_array(self) -> np.ndarray:
        """Spłaszcza cechy liczbowe do jednowymiarowej tablicy o stałej długości.

        Zwraca tablicę ``float64`` o długości :data:`FEATURE_VECTOR_LENGTH`
        (= 42), niezależnej od długości pliku wejściowego (Wymaganie 2.3,
        *Property 3*). Reprezentacja jest wykorzystywana w obliczeniach
        odległości euklidesowej i Mahalanobisa (*Funkcja_Dopasowania*,
        Wymaganie 4.3; ewaluacja obiektywna, Wymaganie 6.1-6.2).

        Układ indeksów (stały):

        =========  ====================================  ========
        Indeksy    Cecha                                 Długość
        =========  ====================================  ========
        0          ``tempo_bpm``                         1
        1..12      ``pitch_class_histogram``             12
        13..37     ``interval_histogram``                25
        38         ``note_density_per_s``                1
        39         ``mean_note_duration_s``              1
        40         ``std_note_duration_s``               1
        41         ``rest_ratio``                        1
        =========  ====================================  ========

        Decyzja o pominięciu pola ``key``: ``key`` jest cechą **kategoryczną**
        (etykieta tonacji). Każde jej zakodowanie liczbowe (np. kod całkowity)
        wprowadzałoby pozorny, muzycznie bezsensowny porządek i fałszowałoby
        odległości (różnica kodów tonacji nie odpowiada odległości muzycznej).
        Ponadto tonacja jest pochodną ``pitch_class_histogram``, więc jej
        liczbowe ujęcie podwajałoby tę samą informację. Dlatego ``key`` jest
        **wyłączone** z reprezentacji wektorowej, a długość pozostaje stała.

        Returns:
            Tablica ``np.ndarray`` o kształcie ``(FEATURE_VECTOR_LENGTH,)`` i
            ``dtype=float64`` (zapisywalna, niezależna kopia danych).
        """
        return np.concatenate(
            (
                np.array([self.tempo_bpm], dtype=np.float64),
                np.asarray(self.pitch_class_histogram, dtype=np.float64),
                np.asarray(self.interval_histogram, dtype=np.float64),
                np.array(
                    [
                        self.note_density_per_s,
                        self.mean_note_duration_s,
                        self.std_note_duration_s,
                        self.rest_ratio,
                    ],
                    dtype=np.float64,
                ),
            )
        )

    @classmethod
    def from_array(cls, array: Any, key: str = "C major") -> "FeatureVector":
        """Rekonstruuje *Wektor_Cech* z reprezentacji wektorowej (odwrotność :meth:`as_array`).

        Metoda jest dokładną odwrotnością :meth:`as_array` w zakresie cech
        liczbowych - rozdziela jednowymiarową tablicę o długości
        :data:`FEATURE_VECTOR_LENGTH` na poszczególne pola, zachowując ten sam
        układ indeksów (tempo, histogram klas wysokości, histogram interwałów,
        gęstość nut, średnia/odchylenie długości nuty, proporcja pauz).

        Pole kategoryczne ``key`` jest **wyłączone** z reprezentacji wektorowej
        (patrz uzasadnienie przy :meth:`as_array`), dlatego musi zostać podane
        osobno; domyślnie przyjmowana jest neutralna tonacja ``"C major"``.
        Metoda jest wykorzystywana m.in. przez
        :meth:`~musicians_style.features.extractor.FeatureExtractor.extract_dataset`
        do zbudowania agregowanych *Wektorów_Cech* (średnia, mediana, odchylenie
        standardowe) z kolumnowych statystyk macierzy cech (Wymaganie 2.6).

        Args:
            array: jednowymiarowa tablica cech liczbowych o długości
                :data:`FEATURE_VECTOR_LENGTH` (układ jak w :meth:`as_array`).
            key: etykieta tonacji przypisywana rekonstruowanemu wektorowi
                (cecha kategoryczna nieobecna w ``array``).

        Returns:
            :class:`FeatureVector` o polach liczbowych odtworzonych z ``array``.

        Raises:
            ValueError: gdy ``array`` nie jest jednowymiarowa lub ma długość inną
                niż :data:`FEATURE_VECTOR_LENGTH`.
        """
        arr = np.asarray(array, dtype=np.float64)
        if arr.shape != (FEATURE_VECTOR_LENGTH,):
            raise ValueError(
                f"from_array oczekuje tablicy o kształcie ({FEATURE_VECTOR_LENGTH},), "
                f"otrzymano {arr.shape}."
            )

        pitch_start = 1
        pitch_end = pitch_start + PITCH_CLASS_BINS
        interval_end = pitch_end + INTERVAL_HISTOGRAM_BINS

        return cls(
            tempo_bpm=float(arr[0]),
            key=key,
            pitch_class_histogram=arr[pitch_start:pitch_end],
            interval_histogram=arr[pitch_end:interval_end],
            note_density_per_s=float(arr[interval_end]),
            mean_note_duration_s=float(arr[interval_end + 1]),
            std_note_duration_s=float(arr[interval_end + 2]),
            rest_ratio=float(arr[interval_end + 3]),
        )

    def __eq__(self, other: object) -> bool:
        """Równość wartościowa: pola skalarne dokładnie, tablice element-wise.

        Pola skalarne (``tempo_bpm``, ``key``, gęstości, ``rest_ratio``)
        porównywane są operatorem ``==``; tablice (``pitch_class_histogram``,
        ``interval_histogram``) przez :func:`_arrays_equal`. Pozwala to
        asertować bit-identyczną równość w teście idempotencji (*Property 2*).
        """
        if other.__class__ is not self.__class__:
            return NotImplemented
        assert isinstance(other, FeatureVector)
        return (
            self.tempo_bpm == other.tempo_bpm
            and self.key == other.key
            and self.note_density_per_s == other.note_density_per_s
            and self.mean_note_duration_s == other.mean_note_duration_s
            and self.std_note_duration_s == other.std_note_duration_s
            and self.rest_ratio == other.rest_ratio
            and _arrays_equal(
                self.pitch_class_histogram, other.pitch_class_histogram
            )
            and _arrays_equal(self.interval_histogram, other.interval_histogram)
        )

    def __hash__(self) -> int:
        """Hash spójny z ``__eq__`` (oparty na bajtach tablic ``float64``).

        Równość pozostaje semantyką nadrzędną; ``__hash__`` umożliwia użycie
        wektorów jako elementów zbiorów/kluczy słowników. Tablice są
        ``float64`` o ustalonym kształcie, więc ``tobytes()`` daje stabilny,
        deterministyczny odcisk.
        """
        return hash(
            (
                self.tempo_bpm,
                self.key,
                self.note_density_per_s,
                self.mean_note_duration_s,
                self.std_note_duration_s,
                self.rest_ratio,
                self.pitch_class_histogram.tobytes(),
                self.interval_histogram.tobytes(),
            )
        )


@dataclass(frozen=True, eq=False)
class AggregatedFeatures:
    """Agregat statystyczny *Wektorów_Cech* *Zbioru_Stylu* (Wymaganie 2.6).

    Atrybuty:
        mean: *Wektor_Cech* średnich wartości cech zbioru.
        median: *Wektor_Cech* median cech zbioru.
        std: *Wektor_Cech* odchyleń standardowych cech zbioru.
        covariance: macierz kowariancji cech liczbowych, kształt
            ``(FEATURE_VECTOR_LENGTH, FEATURE_VECTOR_LENGTH)`` (= 42 x 42),
            zgodna z układem :meth:`FeatureVector.as_array`. Wykorzystywana do
            odległości Mahalanobisa (Wymaganie 4.3).

    ``covariance`` jest kopiowana do tablicy ``float64`` tylko do odczytu i musi
    być kwadratowa o boku :data:`FEATURE_VECTOR_LENGTH`.
    """

    mean: FeatureVector
    median: FeatureVector
    std: FeatureVector
    covariance: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "covariance",
            _as_readonly_float_array(
                self.covariance,
                (FEATURE_VECTOR_LENGTH, FEATURE_VECTOR_LENGTH),
                "covariance",
            ),
        )

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        assert isinstance(other, AggregatedFeatures)
        return (
            self.mean == other.mean
            and self.median == other.median
            and self.std == other.std
            and _arrays_equal(self.covariance, other.covariance)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.mean,
                self.median,
                self.std,
                self.covariance.tobytes(),
            )
        )
