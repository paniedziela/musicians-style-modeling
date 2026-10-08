"""Ewaluator subiektywny - testy odsłuchowe (zadania 11.4-11.5, Wymagania 7.1-7.5).

Moduł przygotowuje materiał do *Testu_Odsłuchowego* (sekcja *Ewaluator* w
``design.md``):

* :meth:`SubjectiveEvaluator.prepare_listening_set` buduje :class:`ListeningSet`
  zawierający co najmniej 10 par ``(Utwór_Wejściowy, Utwór_Wyjściowy)`` wraz z
  fragmentami referencyjnymi *Zbioru_Stylu* (Wymaganie 7.1),
* :meth:`SubjectiveEvaluator.render_to_audio` renderuje plik MIDI do formatu
  audio **WAV (44,1 kHz, 16 bit)** przy użyciu syntezatora **FluidSynth** o
  ujednoliconej konfiguracji (Wymaganie 7.2),
* fragmenty pochodzące z chronionego prawem autorskim *Zbioru_Stylu* są
  ograniczane do **15 sekund** (Wymaganie 7.3),
* :meth:`SubjectiveEvaluator.build_form` generuje serializowalną, deterministyczną
  specyfikację formularza ankietowego (:class:`FormSpec`) - test **ABX** lub
  ankieta **MOS** w skali 1-5 dla podobieństwa stylistycznego i jakości
  muzycznej (Wymaganie 7.4),
* :meth:`SubjectiveEvaluator.analyze_responses` oblicza statystyki opisowe
  (średnia, odchylenie standardowe, przedziały ufności) **bez** przycinania do
  skali ``[1, 5]`` (Wymaganie 7.4) i przeprowadza test istotności wyłącznie, gdy
  liczba respondentów wynosi co najmniej 20 (Wymaganie 7.5).

Lazy import FluidSynth i testowalność
=====================================

Renderowanie audio zależy od biblioteki **FluidSynth** (``pyFluidSynth``,
import jako ``fluidsynth``) oraz od pliku *SoundFont* (``.sf2``), które w
środowisku CI bywają niedostępne. Aby moduł pozostał w pełni testowalny bez
realnego *SoundFontu* ani sprzętu audio:

* zależność FluidSynth jest **importowana leniwie** dopiero w domyślnym
  rendererze (:func:`default_fluidsynth_renderer`), a nie na poziomie modułu;
* :class:`SubjectiveEvaluator` przyjmuje **wstrzykiwalny** ``renderer`` -
  dowolny obiekt wywoływalny ``(midi_path, soundfont, sample_rate) -> np.ndarray``
  zwracający przebieg audio (mono, ``float`` w zakresie ``[-1, 1]``). W testach
  wstrzykiwany jest renderer-zaślepka, dzięki czemu generowanie WAV nie wymaga
  ani realnego ``.sf2``, ani urządzenia dźwiękowego.

Zapis WAV (16 bit PCM, 44,1 kHz) realizowany jest modułem standardowym
``wave`` - bez dodatkowych zależności.
"""

from __future__ import annotations

import math
import warnings
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
from scipy.stats import t as _student_t
from scipy.stats import ttest_1samp

from musicians_style.logging import get_logger

__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "PROTECTED_CLIP_SECONDS",
    "MIN_LISTENING_PAIRS",
    "MOS_SCALE_MIN",
    "MOS_SCALE_MAX",
    "MOS_DIMENSIONS",
    "ABX_CHOICES",
    "SIGNIFICANCE_MIN_RESPONDENTS",
    "DEFAULT_CONFIDENCE_LEVEL",
    "AudioRenderer",
    "FormType",
    "ListeningPair",
    "ListeningSet",
    "FormQuestion",
    "FormSpec",
    "Response",
    "SignificanceResult",
    "MetricStats",
    "SubjectiveReport",
    "SubjectiveEvaluator",
    "default_fluidsynth_renderer",
]

#: Częstotliwość próbkowania eksportu audio (Wymaganie 7.2: 44,1 kHz).
DEFAULT_SAMPLE_RATE: int = 44100

#: Maksymalna długość fragmentu chronionego *Zbioru_Stylu* (Wymaganie 7.3).
PROTECTED_CLIP_SECONDS: int = 15

#: Minimalna liczba par w zestawie odsłuchowym (Wymaganie 7.1).
MIN_LISTENING_PAIRS: int = 10

#: Dolna granica skali ocen MOS (Wymaganie 7.4: skala 1-5).
MOS_SCALE_MIN: int = 1

#: Górna granica skali ocen MOS (Wymaganie 7.4: skala 1-5).
MOS_SCALE_MAX: int = 5

#: Wymiary oceniane w ankiecie MOS (Wymaganie 7.4): podobieństwo stylistyczne
#: oraz jakość muzyczna *Utworu_Wyjściowego*.
MOS_DIMENSIONS: tuple[str, ...] = ("similarity", "quality")

#: Dozwolone odpowiedzi w teście ABX (wskazanie, do którego bodźca - A czy B -
#: bardziej podobny jest bodziec X).
ABX_CHOICES: tuple[str, ...] = ("A", "B")

#: Minimalna liczba respondentów wymagana do przeprowadzenia testu istotności
#: (Wymaganie 7.5: ankieta wypełniona przez co najmniej 20 respondentów).
SIGNIFICANCE_MIN_RESPONDENTS: int = 20

#: Domyślny poziom ufności przedziałów ufności (95%).
DEFAULT_CONFIDENCE_LEVEL: float = 0.95

#: Typ formularza ankietowego (Wymaganie 7.4).
FormType = Literal["ABX", "MOS"]

#: Typ renderera audio: ``(midi_path, soundfont, sample_rate) -> waveform``.
#: ``waveform`` to jednowymiarowa tablica ``float`` (mono) w zakresie ``[-1, 1]``.
AudioRenderer = Callable[[Path, "Path | None", int], np.ndarray]

_log = get_logger("subjective_evaluator")


@dataclass(frozen=True)
class ListeningPair:
    """Para odsłuchowa ``(Utwór_Wejściowy, Utwór_Wyjściowy)`` (Wymaganie 7.1).

    Atrybuty:
        index: numer porządkowy pary w zestawie (od 0).
        input_path: ścieżka *Utworu_Wejściowego* (MIDI).
        output_path: ścieżka *Utworu_Wyjściowego* (MIDI po transferze stylu).
    """

    index: int
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class ListeningSet:
    """Zestaw materiału do *Testu_Odsłuchowego* (Wymaganie 7.1).

    Atrybuty:
        pairs: krotka par ``(Utwór_Wejściowy, Utwór_Wyjściowy)``.
        reference_fragments: krotka ścieżek fragmentów referencyjnych
            *Zbioru_Stylu* (chronione prawem autorskim - renderowane do maks.
            :data:`PROTECTED_CLIP_SECONDS` sekund, Wymaganie 7.3).
        clip_seconds: docelowa maksymalna długość renderowanej próbki w sekundach.
        sample_rate: częstotliwość próbkowania eksportu audio (Hz).
    """

    pairs: tuple[ListeningPair, ...]
    reference_fragments: tuple[Path, ...] = ()
    clip_seconds: int = PROTECTED_CLIP_SECONDS
    sample_rate: int = DEFAULT_SAMPLE_RATE

    @property
    def n_pairs(self) -> int:
        """Liczba par odsłuchowych w zestawie."""
        return len(self.pairs)

    @property
    def meets_minimum(self) -> bool:
        """Czy zestaw spełnia minimum 10 par (Wymaganie 7.1)."""
        return self.n_pairs >= MIN_LISTENING_PAIRS


@dataclass(frozen=True)
class FormQuestion:
    """Pojedyncze pytanie formularza ankietowego (Wymaganie 7.4).

    Reprezentacja jest serializowalna (metoda :meth:`to_dict`) i deterministyczna -
    identyczny :class:`ListeningSet` zawsze generuje identyczne pytania.

    Atrybuty:
        question_id: stabilny identyfikator pytania (np. ``"pair00_similarity"``
            dla MOS lub ``"pair00_abx"`` dla ABX).
        pair_index: numer pary odsłuchowej, której dotyczy pytanie.
        prompt: treść pytania prezentowana respondentowi.
        scale: w testach MOS para ``(min, max)`` skali liczbowej (1-5,
            Wymaganie 7.4); ``None`` dla pytań ABX.
        choices: dozwolone odpowiedzi wyboru (np. ``("A", "B")`` dla ABX);
            ``None`` dla pytań ocenianych w skali.
        dimension: oceniany wymiar w MOS (``"similarity"`` / ``"quality"``);
            ``None`` dla ABX.
        stimuli: mapowanie etykiet bodźców (np. ``{"A": ..., "B": ..., "X": ...}``
            w ABX) na ścieżki plików audio/MIDI - dokumentuje, które próbki
            należy odtworzyć.
    """

    question_id: str
    pair_index: int
    prompt: str
    scale: tuple[int, int] | None = None
    choices: tuple[str, ...] | None = None
    dimension: str | None = None
    stimuli: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializuje pytanie do słownika (gotowego do zapisu jako JSON)."""
        return {
            "question_id": self.question_id,
            "pair_index": self.pair_index,
            "prompt": self.prompt,
            "scale": list(self.scale) if self.scale is not None else None,
            "choices": list(self.choices) if self.choices is not None else None,
            "dimension": self.dimension,
            "stimuli": dict(self.stimuli),
        }


@dataclass(frozen=True)
class FormSpec:
    """Specyfikacja formularza ankietowego *Testu_Odsłuchowego* (Wymaganie 7.4).

    Struktura serializowalna i deterministyczna, opisująca komplet pytań dla
    testu ABX lub ankiety MOS (skala 1-5). Umożliwia respondentom ocenę
    podobieństwa stylistycznego oraz jakości muzycznej *Utworu_Wyjściowego*.

    Atrybuty:
        form_type: rodzaj formularza - ``"ABX"`` lub ``"MOS"`` (Wymaganie 7.4).
        questions: krotka pytań :class:`FormQuestion` w deterministycznej
            kolejności (rosnąco wg numeru pary, a w MOS wg wymiaru).
        scale: para ``(min, max)`` skali ocen dla MOS (1-5); ``None`` dla ABX.
        n_pairs: liczba par odsłuchowych, na podstawie których zbudowano
            formularz.
    """

    form_type: FormType
    questions: tuple[FormQuestion, ...]
    scale: tuple[int, int] | None = None
    n_pairs: int = 0

    @property
    def n_questions(self) -> int:
        """Liczba pytań w formularzu."""
        return len(self.questions)

    def to_dict(self) -> dict[str, Any]:
        """Serializuje specyfikację formularza do słownika (JSON)."""
        return {
            "form_type": self.form_type,
            "scale": list(self.scale) if self.scale is not None else None,
            "n_pairs": self.n_pairs,
            "n_questions": self.n_questions,
            "questions": [q.to_dict() for q in self.questions],
        }


@dataclass(frozen=True)
class Response:
    """Pojedyncza odpowiedź respondenta na pytanie formularza (Wymaganie 7.5).

    Atrybuty:
        respondent_id: identyfikator respondenta (dowolny, używany wyłącznie do
            zliczania unikalnych respondentów).
        question_id: identyfikator pytania (zgodny z :class:`FormQuestion`).
        value: udzielona odpowiedź - dla MOS ocena liczbowa (typowo 1-5, jednak
            wartości spoza skali są dopuszczalne i nie są przycinane,
            Wymaganie 7.4); dla ABX poprawność wskazania kodowana jako 1.0
            (trafienie) lub 0.0 (pomyłka).
        dimension: oceniany wymiar (``"similarity"``/``"quality"`` dla MOS,
            ``"abx_accuracy"`` dla ABX); gdy ``None`` przyjmowany jest wymiar
            domyślny przy agregacji.
    """

    respondent_id: str
    question_id: str
    value: float
    dimension: str | None = None


@dataclass(frozen=True)
class SignificanceResult:
    """Wynik testu istotności odpowiedzi ankiety (Wymaganie 7.5).

    Atrybuty:
        test: nazwa testu (``"t-test-1samp"`` - jednopróbkowy test t-Studenta
            względem wartości odniesienia).
        statistic: wartość statystyki testowej.
        p_value: uzyskana wartość p.
        alpha: przyjęty poziom istotności.
        significant: ``True`` gdy ``p_value < alpha``.
        popmean: wartość odniesienia (hipoteza zerowa), względem której badana
            jest średnia ocen (np. środek skali MOS lub poziom losowy ABX).
    """

    test: str
    statistic: float
    p_value: float
    alpha: float
    significant: bool
    popmean: float

    def to_dict(self) -> dict[str, Any]:
        """Serializuje wynik testu do słownika (JSON)."""
        return {
            "test": self.test,
            "statistic": _json_safe_float(self.statistic),
            "p_value": _json_safe_float(self.p_value),
            "alpha": self.alpha,
            "significant": self.significant,
            "popmean": _json_safe_float(self.popmean),
        }


@dataclass(frozen=True)
class MetricStats:
    """Statystyki opisowe ocen jednego wymiaru ankiety (Wymaganie 7.4).

    KRYTYCZNE (Wymaganie 7.4): wartości ``mean`` oraz granice przedziału ufności
    ``ci_low``/``ci_high`` są **surowymi** wynikami obliczeń i **nie są
    przycinane** do skali ``[1, 5]``. Mogą zatem wykraczać poza granice skali.

    Atrybuty:
        dimension: nazwa wymiaru (np. ``"similarity"``, ``"quality"``,
            ``"abx_accuracy"``).
        n: liczba ocen wziętych do obliczeń.
        mean: średnia arytmetyczna ocen (bez przycinania).
        std: odchylenie standardowe próby (``ddof=1``; ``0`` dla pojedynczej
            oceny).
        ci_low: dolna granica przedziału ufności średniej (bez przycinania).
        ci_high: górna granica przedziału ufności średniej (bez przycinania).
        confidence_level: poziom ufności przedziału (np. 0,95).
    """

    dimension: str
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float
    confidence_level: float

    def to_dict(self) -> dict[str, Any]:
        """Serializuje statystyki do słownika (JSON)."""
        return {
            "dimension": self.dimension,
            "n": self.n,
            "mean": _json_safe_float(self.mean),
            "std": _json_safe_float(self.std),
            "ci_low": _json_safe_float(self.ci_low),
            "ci_high": _json_safe_float(self.ci_high),
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True)
class SubjectiveReport:
    """Raport analizy odpowiedzi ankiety subiektywnej (Wymagania 7.4, 7.5).

    Atrybuty:
        n_respondents: liczba unikalnych respondentów, którzy wypełnili ankietę.
        n_responses: łączna liczba odpowiedzi wziętych do analizy.
        metrics: statystyki opisowe (średnia, odchylenie, przedział ufności)
            per wymiar (Wymaganie 7.4) - **bez** przycinania do skali ``[1, 5]``.
        significance: wynik testu istotności (Wymaganie 7.5) - obecny wyłącznie
            gdy ``n_respondents >= 20``; w przeciwnym razie ``None``.
        note: opcjonalna nota diagnostyczna (np. informacja o pominięciu testu
            istotności przy zbyt małej liczbie respondentów).
    """

    n_respondents: int
    n_responses: int
    metrics: dict[str, MetricStats]
    significance: SignificanceResult | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializuje raport do słownika gotowego do zapisu jako JSON."""
        return {
            "n_respondents": self.n_respondents,
            "n_responses": self.n_responses,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "significance": (
                None if self.significance is None else self.significance.to_dict()
            ),
            "note": self.note,
        }


def _json_safe_float(value: float) -> float | None:
    """Zwraca ``value`` jeśli skończone, w przeciwnym razie ``None``.

    Wartości ``NaN``/``±inf`` nie są poprawnym JSON-em w trybie ścisłym, dlatego
    przy serializacji raportów subiektywnych zastępujemy je ``null``.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def default_fluidsynth_renderer(
    midi_path: Path,
    soundfont: Path | None,
    sample_rate: int,
) -> np.ndarray:
    """Domyślny renderer audio oparty na FluidSynth (Wymaganie 7.2).

    Zależności (``pretty_midi`` + ``fluidsynth``/``pyFluidSynth``) są importowane
    **leniwie** wewnątrz funkcji, aby sam moduł pozostał importowalny i testowalny
    w środowisku bez tych bibliotek lub bez *SoundFontu*.

    Args:
        midi_path: ścieżka pliku MIDI do zsyntetyzowania.
        soundfont: ścieżka pliku *SoundFont* (``.sf2``); wymagana dla realnej
            syntezy.
        sample_rate: częstotliwość próbkowania (Hz).

    Returns:
        Jednowymiarowa tablica ``float`` (mono) - przebieg audio.

    Raises:
        RuntimeError: gdy ``pretty_midi`` nie jest dostępny.
        FileNotFoundError: gdy podany *SoundFont* nie istnieje.
    """
    if soundfont is None or not Path(soundfont).exists():
        raise FileNotFoundError(
            f"SoundFont wymagany do syntezy FluidSynth nie istnieje: {soundfont!r}."
        )

    try:
        import pretty_midi  # lazy import - patrz docstring modułu
    except ImportError as exc:  # pragma: no cover - zależne od środowiska
        raise RuntimeError(
            "Renderowanie audio wymaga pakietu 'pretty_midi' oraz 'pyFluidSynth'. "
            "Zainstaluj zależności lub wstrzyknij własny renderer."
        ) from exc

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    # pretty_midi.fluidsynth() leniwie korzysta z biblioteki fluidsynth.
    waveform = pm.fluidsynth(fs=int(sample_rate), sf2_path=str(soundfont))
    return np.asarray(waveform, dtype=np.float64).reshape(-1)


class SubjectiveEvaluator:
    """Przygotowanie *Testu_Odsłuchowego* i eksport audio (Wymagania 7.1-7.3).

    Ewaluator jest konfigurowany ujednoliconym *SoundFontem* i częstotliwością
    próbkowania (Wymaganie 7.2). Renderer audio jest wstrzykiwalny, co umożliwia
    testowanie bez realnego *SoundFontu* (patrz docstring modułu).
    """

    def __init__(
        self,
        *,
        soundfont: Path | str | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        clip_seconds: int = PROTECTED_CLIP_SECONDS,
        renderer: AudioRenderer | None = None,
    ) -> None:
        """Inicjalizuje ewaluator subiektywny.

        Args:
            soundfont: ścieżka ujednoliconego *SoundFontu* (``.sf2``) używanego
                przy renderowaniu (Wymaganie 7.2). Może być ``None``, gdy
                wstrzyknięto renderer-zaślepkę.
            sample_rate: częstotliwość próbkowania eksportu audio (domyślnie
                44 100 Hz, Wymaganie 7.2).
            clip_seconds: domyślna maksymalna długość renderowanej próbki w
                sekundach (Wymaganie 7.3).
            renderer: opcjonalny, wstrzykiwalny renderer audio
                (:data:`AudioRenderer`); gdy ``None`` używany jest
                :func:`default_fluidsynth_renderer`.

        Raises:
            ValueError: gdy ``sample_rate`` lub ``clip_seconds`` jest niedodatnie.
        """
        if sample_rate <= 0:
            raise ValueError(f"sample_rate musi być dodatnie, otrzymano {sample_rate}.")
        if clip_seconds <= 0:
            raise ValueError(f"clip_seconds musi być dodatnie, otrzymano {clip_seconds}.")
        self.soundfont = Path(soundfont) if soundfont is not None else None
        self.sample_rate = int(sample_rate)
        self.clip_seconds = int(clip_seconds)
        self._renderer: AudioRenderer = (
            renderer if renderer is not None else default_fluidsynth_renderer
        )

    # -- przygotowanie zestawu odsłuchowego ---------------------------------

    def prepare_listening_set(
        self,
        pairs: Sequence[tuple[Path | str, Path | str]],
        reference_fragments: Sequence[Path | str] = (),
        n_pairs: int = MIN_LISTENING_PAIRS,
    ) -> ListeningSet:
        """Buduje :class:`ListeningSet` z par i fragmentów referencyjnych (Wymaganie 7.1).

        Z dostarczonej kolekcji par wybieranych jest pierwszych ``n_pairs``
        (zachowując kolejność). Zgodnie z Wymaganiem 7.1 zestaw powinien
        zawierać co najmniej 10 par - gdy dostępnych jest mniej niż ``n_pairs``
        (lub mniej niż :data:`MIN_LISTENING_PAIRS`), w logu zapisywane jest
        ostrzeżenie, a zestaw jest budowany z dostępnych par.

        Args:
            pairs: kolekcja par ścieżek ``(input_midi, output_midi)``.
            reference_fragments: ścieżki fragmentów referencyjnych *Zbioru_Stylu*.
            n_pairs: docelowa liczba par w zestawie (domyślnie 10).

        Returns:
            :class:`ListeningSet` z parami i fragmentami referencyjnymi.

        Raises:
            ValueError: gdy ``n_pairs`` jest niedodatnie lub ``pairs`` jest puste.
        """
        if n_pairs <= 0:
            raise ValueError(f"n_pairs musi być dodatnie, otrzymano {n_pairs}.")
        pair_list = list(pairs)
        if not pair_list:
            raise ValueError(
                "Nie można przygotować zestawu odsłuchowego z pustej listy par."
            )

        if len(pair_list) < n_pairs:
            _log.warning(
                "dostępnych par mniej niż żądana liczność zestawu odsłuchowego",
                available=len(pair_list),
                requested=n_pairs,
            )
        selected = pair_list[:n_pairs]

        if len(selected) < MIN_LISTENING_PAIRS:
            _log.warning(
                "zestaw odsłuchowy nie spełnia minimum 10 par (Wymaganie 7.1)",
                n_pairs=len(selected),
                minimum=MIN_LISTENING_PAIRS,
            )

        listening_pairs = tuple(
            ListeningPair(
                index=i,
                input_path=Path(inp),
                output_path=Path(out),
            )
            for i, (inp, out) in enumerate(selected)
        )
        references = tuple(Path(p) for p in reference_fragments)

        _log.info(
            "przygotowano zestaw odsłuchowy",
            n_pairs=len(listening_pairs),
            n_references=len(references),
            clip_seconds=self.clip_seconds,
        )
        return ListeningSet(
            pairs=listening_pairs,
            reference_fragments=references,
            clip_seconds=self.clip_seconds,
            sample_rate=self.sample_rate,
        )

    # -- eksport audio -------------------------------------------------------

    def render_to_audio(
        self,
        midi_path: Path | str,
        soundfont: Path | str | None = None,
        clip_seconds: int | None = None,
        *,
        output_path: Path | str | None = None,
        protected: bool = False,
    ) -> Path:
        """Renderuje plik MIDI do WAV (44,1 kHz, 16 bit) przez FluidSynth (Wymaganie 7.2).

        Przebieg audio jest pozyskiwany z wstrzykniętego (lub domyślnego)
        renderera, przycinany do długości ``clip_seconds`` i zapisywany jako WAV
        16-bitowy mono o częstotliwości :attr:`sample_rate`.

        Dla próbek chronionych (``protected=True``, fragmenty *Zbioru_Stylu*)
        długość jest dodatkowo ograniczana do :data:`PROTECTED_CLIP_SECONDS`
        (15 s, Wymaganie 7.3) niezależnie od ``clip_seconds``.

        Args:
            midi_path: ścieżka pliku MIDI do renderowania.
            soundfont: *SoundFont* (``.sf2``); gdy ``None`` używany jest
                skonfigurowany :attr:`soundfont`.
            clip_seconds: maksymalna długość próbki w sekundach; gdy ``None``
                używana jest wartość :attr:`clip_seconds`.
            output_path: docelowa ścieżka WAV; gdy ``None`` tworzona jest obok
                pliku MIDI ze zmienionym rozszerzeniem na ``.wav``.
            protected: czy próbka pochodzi z chronionego *Zbioru_Stylu*
                (wymusza limit 15 s, Wymaganie 7.3).

        Returns:
            Ścieżka zapisanego pliku WAV.

        Raises:
            ValueError: gdy ``clip_seconds`` jest niedodatnie.
        """
        midi = Path(midi_path)
        sf = Path(soundfont) if soundfont is not None else self.soundfont

        limit = self.clip_seconds if clip_seconds is None else int(clip_seconds)
        if limit <= 0:
            raise ValueError(f"clip_seconds musi być dodatnie, otrzymano {limit}.")
        if protected:
            limit = min(limit, PROTECTED_CLIP_SECONDS)

        waveform = self._renderer(midi, sf, self.sample_rate)
        waveform = np.asarray(waveform, dtype=np.float64).reshape(-1)

        max_samples = limit * self.sample_rate
        if waveform.shape[0] > max_samples:
            waveform = waveform[:max_samples]

        target = (
            Path(output_path)
            if output_path is not None
            else midi.with_suffix(".wav")
        )
        self._write_wav_16bit(target, waveform)

        _log.info(
            "wyrenderowano próbkę audio",
            midi=str(midi),
            wav=str(target),
            sample_rate=self.sample_rate,
            clip_seconds=limit,
            protected=protected,
        )
        return target

    def render_listening_set(
        self,
        listening_set: ListeningSet,
        output_dir: Path | str,
        soundfont: Path | str | None = None,
    ) -> dict[str, list[Path]]:
        """Renderuje cały zestaw odsłuchowy do plików WAV.

        Pary ``(input, output)`` renderowane są w pełnej długości ``clip_seconds``,
        a fragmenty referencyjne *Zbioru_Stylu* z limitem 15 s (Wymaganie 7.3).

        Args:
            listening_set: zestaw odsłuchowy z :meth:`prepare_listening_set`.
            output_dir: katalog docelowy plików WAV (tworzony, gdy nie istnieje).
            soundfont: *SoundFont*; gdy ``None`` używany jest skonfigurowany.

        Returns:
            Słownik ze ścieżkami WAV pod kluczami ``"inputs"``, ``"outputs"``,
            ``"references"``.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sf = Path(soundfont) if soundfont is not None else self.soundfont

        inputs: list[Path] = []
        outputs: list[Path] = []
        references: list[Path] = []

        for pair in listening_set.pairs:
            inputs.append(
                self.render_to_audio(
                    pair.input_path,
                    sf,
                    listening_set.clip_seconds,
                    output_path=out_dir / f"pair{pair.index:02d}_input.wav",
                )
            )
            outputs.append(
                self.render_to_audio(
                    pair.output_path,
                    sf,
                    listening_set.clip_seconds,
                    output_path=out_dir / f"pair{pair.index:02d}_output.wav",
                )
            )

        for i, ref in enumerate(listening_set.reference_fragments):
            references.append(
                self.render_to_audio(
                    ref,
                    sf,
                    PROTECTED_CLIP_SECONDS,
                    output_path=out_dir / f"reference{i:02d}.wav",
                    protected=True,  # Wymaganie 7.3: limit 15 s
                )
            )

        return {"inputs": inputs, "outputs": outputs, "references": references}

    # -- zapis WAV -----------------------------------------------------------

    def _write_wav_16bit(self, path: Path, waveform: np.ndarray) -> None:
        """Zapisuje przebieg audio jako WAV 16-bit PCM mono (Wymaganie 7.2).

        Wartości ``float`` są przycinane do ``[-1, 1]`` i kwantowane do 16 bitów
        ze znakiem (little-endian). Częstotliwość ramki to :attr:`sample_rate`.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        clipped = np.clip(waveform, -1.0, 1.0)
        pcm16 = np.round(clipped * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)  # mono
            wav.setsampwidth(2)  # 16 bit
            wav.setframerate(self.sample_rate)  # 44,1 kHz
            wav.writeframes(pcm16.tobytes())

    # -- formularz ankietowy (Wymaganie 7.4) --------------------------------

    def build_form(
        self,
        listening_set: ListeningSet,
        form_type: FormType,
    ) -> FormSpec:
        """Buduje specyfikację formularza ankietowego *Testu_Odsłuchowego* (Wymaganie 7.4).

        Dla ``form_type == "MOS"`` generowane są pytania z oceną w skali 1-5 dla
        dwóch wymiarów: podobieństwa stylistycznego (``"similarity"``) oraz
        jakości muzycznej (``"quality"``) każdej pary odsłuchowej. Dla
        ``form_type == "ABX"`` generowane jest pytanie ABX na parę: respondent
        wskazuje, czy bodziec referencyjny ``X`` jest bardziej podobny do ``A``
        czy do ``B``.

        Generowanie jest **deterministyczne**: ten sam :class:`ListeningSet`
        zawsze daje identyczną (co do kolejności i treści) :class:`FormSpec`.
        Pytania są uporządkowane rosnąco wg numeru pary; w MOS w obrębie pary wg
        kolejności wymiarów z :data:`MOS_DIMENSIONS`.

        Args:
            listening_set: zestaw odsłuchowy z :meth:`prepare_listening_set`.
            form_type: rodzaj formularza - ``"ABX"`` lub ``"MOS"``.

        Returns:
            :class:`FormSpec` z deterministyczną listą pytań.

        Raises:
            ValueError: gdy ``form_type`` jest spoza ``{"ABX", "MOS"}`` lub gdy
                zestaw odsłuchowy nie zawiera żadnej pary.
        """
        if form_type not in ("ABX", "MOS"):
            raise ValueError(
                f"Nieobsługiwany typ formularza: {form_type!r} "
                "(dozwolone 'ABX', 'MOS')."
            )
        if listening_set.n_pairs == 0:
            raise ValueError(
                "Nie można zbudować formularza z pustego zestawu odsłuchowego."
            )

        if form_type == "MOS":
            spec = self._build_mos_form(listening_set)
        else:
            spec = self._build_abx_form(listening_set)

        _log.info(
            "zbudowano formularz ankietowy",
            form_type=form_type,
            n_pairs=listening_set.n_pairs,
            n_questions=spec.n_questions,
        )
        return spec

    def _build_mos_form(self, listening_set: ListeningSet) -> FormSpec:
        """Buduje formularz MOS (skala 1-5) dla podobieństwa i jakości (Wymaganie 7.4)."""
        scale = (MOS_SCALE_MIN, MOS_SCALE_MAX)
        prompts = {
            "similarity": (
                "W skali od {lo} do {hi} oceń, na ile Utwór_Wyjściowy jest "
                "stylistycznie podobny do utworów artysty docelowego."
            ).format(lo=MOS_SCALE_MIN, hi=MOS_SCALE_MAX),
            "quality": (
                "W skali od {lo} do {hi} oceń ogólną jakość muzyczną "
                "Utworu_Wyjściowego."
            ).format(lo=MOS_SCALE_MIN, hi=MOS_SCALE_MAX),
        }
        questions: list[FormQuestion] = []
        for pair in listening_set.pairs:
            for dimension in MOS_DIMENSIONS:
                questions.append(
                    FormQuestion(
                        question_id=f"pair{pair.index:02d}_{dimension}",
                        pair_index=pair.index,
                        prompt=prompts[dimension],
                        scale=scale,
                        choices=None,
                        dimension=dimension,
                        stimuli={
                            "input": str(pair.input_path),
                            "output": str(pair.output_path),
                        },
                    )
                )
        return FormSpec(
            form_type="MOS",
            questions=tuple(questions),
            scale=scale,
            n_pairs=listening_set.n_pairs,
        )

    def _build_abx_form(self, listening_set: ListeningSet) -> FormSpec:
        """Buduje formularz ABX (wskazanie A/B) dla każdej pary (Wymaganie 7.4)."""
        prompt = (
            "Odsłuchaj bodźce A, B oraz X. Wskaż, do którego z bodźców (A czy B) "
            "bardziej podobny jest bodziec X."
        )
        questions: list[FormQuestion] = []
        for pair in listening_set.pairs:
            # A = Utwór_Wejściowy, B = Utwór_Wyjściowy; X (bodziec ukryty) jest
            # przypisywany podczas administrowania ankietą - tu dokumentujemy
            # jedynie kandydatów A/B w sposób deterministyczny.
            questions.append(
                FormQuestion(
                    question_id=f"pair{pair.index:02d}_abx",
                    pair_index=pair.index,
                    prompt=prompt,
                    scale=None,
                    choices=ABX_CHOICES,
                    dimension="abx_accuracy",
                    stimuli={
                        "A": str(pair.input_path),
                        "B": str(pair.output_path),
                    },
                )
            )
        return FormSpec(
            form_type="ABX",
            questions=tuple(questions),
            scale=None,
            n_pairs=listening_set.n_pairs,
        )

    # -- analiza odpowiedzi (Wymagania 7.4, 7.5) ----------------------------

    def analyze_responses(
        self,
        responses: Sequence[Response],
        *,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        popmean: float | None = None,
        alpha: float = 0.05,
    ) -> SubjectiveReport:
        """Analizuje odpowiedzi ankiety: statystyki opisowe i test istotności.

        Dla każdego wymiaru (``dimension``) obliczane są: średnia, odchylenie
        standardowe próby (``ddof=1``) oraz przedział ufności średniej oparty na
        rozkładzie t-Studenta.

        KRYTYCZNE (Wymaganie 7.4): średnia oraz granice przedziału ufności są
        **surowymi** wartościami obliczeń i **nie są przycinane** do skali
        ``[1, 5]`` - mogą wykraczać poza granice skali.

        Test istotności (Wymaganie 7.5) jest przeprowadzany **wyłącznie**, gdy
        liczba unikalnych respondentów wynosi co najmniej
        :data:`SIGNIFICANCE_MIN_RESPONDENTS` (20). W przeciwnym razie zwracane są
        tylko statystyki opisowe, ``significance is None``, a w logu zapisywana
        jest nota o pominięciu testu.

        Args:
            responses: kolekcja odpowiedzi respondentów (:class:`Response`).
            confidence_level: poziom ufności przedziałów (domyślnie 0,95).
            popmean: wartość odniesienia hipotezy zerowej testu istotności; gdy
                ``None`` przyjmowany jest środek skali MOS
                (``(MOS_SCALE_MIN + MOS_SCALE_MAX) / 2``).
            alpha: poziom istotności testu (domyślnie 0,05).

        Returns:
            :class:`SubjectiveReport` ze statystykami opisowymi i - przy
            wystarczającej liczbie respondentów - wynikiem testu istotności.

        Raises:
            ValueError: gdy ``responses`` jest puste lub ``confidence_level``
                jest spoza przedziału ``(0, 1)``.
        """
        response_list = list(responses)
        if not response_list:
            raise ValueError("Brak odpowiedzi do analizy (pusta kolekcja).")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(
                "confidence_level musi należeć do (0, 1), otrzymano "
                f"{confidence_level}."
            )

        n_respondents = len({r.respondent_id for r in response_list})
        ref_mean = (
            popmean
            if popmean is not None
            else (MOS_SCALE_MIN + MOS_SCALE_MAX) / 2.0
        )

        # Grupowanie ocen per wymiar (zachowanie deterministycznej kolejności
        # pierwszego wystąpienia wymiaru w danych wejściowych).
        grouped: dict[str, list[float]] = {}
        for resp in response_list:
            dimension = resp.dimension if resp.dimension is not None else "overall"
            grouped.setdefault(dimension, []).append(float(resp.value))

        metrics = {
            dimension: self._metric_stats(dimension, values, confidence_level)
            for dimension, values in grouped.items()
        }

        # Wymaganie 7.5: test istotności tylko dla n_respondents >= 20.
        if n_respondents >= SIGNIFICANCE_MIN_RESPONDENTS:
            all_values = np.asarray(
                [float(r.value) for r in response_list], dtype=np.float64
            )
            significance = self._significance_test(all_values, ref_mean, alpha)
            note = None
            _log.info(
                "analiza odpowiedzi ankiety: przeprowadzono test istotności",
                n_respondents=n_respondents,
                n_responses=len(response_list),
                test=significance.test,
                p_value=_json_safe_float(significance.p_value),
                significant=significance.significant,
            )
        else:
            significance = None
            note = (
                "Pominięto test istotności - liczba respondentów "
                f"({n_respondents}) jest mniejsza niż wymagane minimum "
                f"{SIGNIFICANCE_MIN_RESPONDENTS} (Wymaganie 7.5); "
                "raport zawiera wyłącznie statystyki opisowe."
            )
            _log.warning(
                "niewystarczająca liczba respondentów do testu istotności - "
                "raport zawiera wyłącznie statystyki opisowe",
                n_respondents=n_respondents,
                min_respondents=SIGNIFICANCE_MIN_RESPONDENTS,
            )

        return SubjectiveReport(
            n_respondents=n_respondents,
            n_responses=len(response_list),
            metrics=metrics,
            significance=significance,
            note=note,
        )

    @staticmethod
    def _metric_stats(
        dimension: str,
        values: Sequence[float],
        confidence_level: float,
    ) -> MetricStats:
        """Oblicza statystyki opisowe wymiaru BEZ przycinania do skali (Wymaganie 7.4).

        Średnia, odchylenie standardowe (``ddof=1``) i przedział ufności średniej
        (rozkład t-Studenta) są raportowane jako surowe wartości obliczeń.
        """
        arr = np.asarray(values, dtype=np.float64)
        n = int(arr.size)
        mean = float(arr.mean())

        if n < 2:
            # Odchylenie próby i przedział ufności nieokreślone dla n < 2:
            # std = 0, a przedział degeneruje się do punktu (bez przycinania).
            return MetricStats(
                dimension=dimension,
                n=n,
                mean=mean,
                std=0.0,
                ci_low=mean,
                ci_high=mean,
                confidence_level=confidence_level,
            )

        std = float(arr.std(ddof=1))
        sem = std / math.sqrt(n)
        # Dwustronny kwantyl rozkładu t-Studenta dla n-1 stopni swobody.
        t_crit = float(_student_t.ppf(0.5 + confidence_level / 2.0, df=n - 1))
        margin = t_crit * sem
        # Wymaganie 7.4: BEZ przycinania granic do [1, 5].
        return MetricStats(
            dimension=dimension,
            n=n,
            mean=mean,
            std=std,
            ci_low=mean - margin,
            ci_high=mean + margin,
            confidence_level=confidence_level,
        )

    @staticmethod
    def _significance_test(
        values: np.ndarray,
        popmean: float,
        alpha: float,
    ) -> SignificanceResult:
        """Jednopróbkowy test t-Studenta średniej ocen względem ``popmean`` (Wymaganie 7.5).

        Bada, czy średnia ocen istotnie różni się od wartości odniesienia
        (np. środka skali MOS). Obsługuje przypadki degenerowane (zerowa
        wariancja) bez zgłaszania wyjątku.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ttest_1samp(values, popmean)
        statistic = float(np.asarray(result.statistic).reshape(-1)[0])
        p_value = float(np.asarray(result.pvalue).reshape(-1)[0])

        if not math.isfinite(p_value):
            # Zerowa wariancja: gdy średnia == popmean -> brak różnicy (p=1),
            # w przeciwnym razie różnica idealnie spójna (p -> 0).
            if math.isclose(float(values.mean()), popmean):
                statistic, p_value = 0.0, 1.0
            else:
                p_value = 0.0

        significant = bool(math.isfinite(p_value) and p_value < alpha)
        return SignificanceResult(
            test="t-test-1samp",
            statistic=statistic,
            p_value=p_value,
            alpha=float(alpha),
            significant=significant,
            popmean=float(popmean),
        )
