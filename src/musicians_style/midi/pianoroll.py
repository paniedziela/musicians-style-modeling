"""Konwerter Pianoroll - pomocnicza reprezentacja *Modelu_GAN* (Wymagania 5.3, 5.7).

Moduł implementuje :class:`Pianoroll` - dwukierunkowy konwerter pomiędzy
:class:`~musicians_style.midi.types.InternalRepr` (lista zdarzeń) a binarną
macierzą *pianoroll* ``[T × P]`` (czas × wysokość) używaną jako wejście
generatywnej sieci neuronowej (StarGAN/CycleGAN), zgodnie z reprezentacją z
pracy [arXiv:1809.07575](https://arxiv.org/abs/1809.07575).

Charakterystyka reprezentacji
-----------------------------
* **Oś czasu (``T``)**: dyskretyzowana z krokiem szesnastki (*16th note*).
  Liczba kroków na ćwierćnutę wynika z konfiguracji
  (``features.pianoroll_steps_per_beat``, domyślnie ``4`` → krok = szesnastka).
  Długość okna ``window_steps`` (domyślnie ``64`` - cztery takty 4/4) ustala
  stały wymiar czasowy oczekiwany przez *Model_GAN*.
* **Oś wysokości (``P``)**: zakres wysokości z konfiguracji
  (``features.pitch_range``, domyślnie ``[24, 108]`` → 84 wysokości C1..B7).
  Dolna granica jest **włączna**, górna **wyłączna**: ``P = pitch_high - pitch_low``.

Stratność i rola *template* (Wymaganie 5.7)
-------------------------------------------
Konwersja ``InternalRepr → pianoroll`` jest **stratna** - binarna macierz nie
przechowuje *velocity*, kanałów MIDI, zdarzeń meta (tempo, metrum, tonacja) ani
mikrorytmiki poniżej rozdzielczości kroku. Dlatego rekonstrukcja
``to_internal`` przyjmuje *template* (zwykle *Utwór_Wejściowy*), z którego
odtwarza:

* ``ticks_per_beat``, ``smf_format`` oraz wszystkie zdarzenia ``meta``
  (tempo, metrum, tonacja, zmiany programu) - co realizuje wymóg zachowania
  tempa i metryki *Utworu_Wyjściowego* (Wymaganie 5.7),
* *velocity* i kanał MIDI każdej zrekonstruowanej nuty - dopasowane do
  najbliższego zdarzenia nutowego *template* o tej samej wysokości, a w braku
  dopasowania zastępowane wartościami reprezentatywnymi (mediana *velocity*,
  dominujący kanał) wyznaczonymi z *template*.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from .types import InternalRepr, NoteEvent, event_key

__all__ = ["Pianoroll"]

# Domyślny zakres wysokości (Wymaganie 5.3, sekcja Data Models design.md):
# [24, 108) → 84 wysokości od C1 (MIDI 24) do B7 (MIDI 107).
_DEFAULT_PITCH_RANGE = (24, 108)
# Domyślna rozdzielczość: 4 kroki na ćwierćnutę == krok szesnastki.
_DEFAULT_STEPS_PER_BEAT = 4
# Domyślna długość okna czasowego (kroki) - cztery takty 4/4.
_DEFAULT_WINDOW_STEPS = 64
# Domyślna granica binaryzacji wyjścia sieci (sigmoid → {0, 1}).
_DEFAULT_THRESHOLD = 0.5
# Wartości zastępcze, gdy *template* nie dostarcza informacji.
_FALLBACK_VELOCITY = 64
_FALLBACK_CHANNEL = 0


class Pianoroll:
    """Dwukierunkowy konwerter *Reprezentacja_Wewnętrzna* ↔ macierz pianoroll.

    Instancja jest bezstanowa i bezpieczna do współdzielenia. Parametry
    geometrii (zakres wysokości, rozdzielczość, domyślna długość okna) są
    ustalane w konstruktorze, dzięki czemu jedna instancja opisuje spójny
    kontrakt wymiarów dla *Modelu_GAN*.

    Args:
        pitch_range: para ``(low, high)`` - dolna granica włączna, górna
            wyłączna. Liczba wysokości ``P = high - low``.
        steps_per_beat: liczba kroków czasowych na ćwierćnutę (``4`` → szesnastka).
        window_steps: domyślna liczba kroków czasowych ``T`` macierzy pianoroll.
    """

    def __init__(
        self,
        pitch_range: tuple[int, int] = _DEFAULT_PITCH_RANGE,
        steps_per_beat: int = _DEFAULT_STEPS_PER_BEAT,
        window_steps: int = _DEFAULT_WINDOW_STEPS,
    ) -> None:
        pitch_low, pitch_high = int(pitch_range[0]), int(pitch_range[1])
        if pitch_high <= pitch_low:
            raise ValueError(
                "pitch_range musi spełniać low < high, otrzymano: "
                f"({pitch_low}, {pitch_high})."
            )
        if steps_per_beat <= 0:
            raise ValueError(
                f"steps_per_beat musi być dodatnie, otrzymano: {steps_per_beat!r}."
            )
        if window_steps <= 0:
            raise ValueError(
                f"window_steps musi być dodatnie, otrzymano: {window_steps!r}."
            )
        self.pitch_low = pitch_low
        self.pitch_high = pitch_high
        self.steps_per_beat = int(steps_per_beat)
        self.window_steps = int(window_steps)

    @property
    def n_pitches(self) -> int:
        """Liczba wysokości ``P`` (szerokość osi wysokości macierzy pianoroll)."""
        return self.pitch_high - self.pitch_low

    # -- API publiczne -------------------------------------------------------

    def from_internal(
        self, repr_: InternalRepr, window_steps: int | None = None
    ) -> np.ndarray:
        """Konwertuje *Reprezentację_Wewnętrzną* na binarną macierz ``[T × P]``.

        Każda nuta zostaje skwantowana do kroków szesnastkowych i zaznaczona
        wartością ``1.0`` we wszystkich krokach, w których brzmi. Nuty (lub ich
        fragmenty) wykraczające poza okno ``window_steps`` są przycinane;
        wysokości spoza ``pitch_range`` są pomijane (reprezentacja jest celowo
        stratna - zob. docstring modułu).

        Args:
            repr_: *Reprezentacja_Wewnętrzna* źródłowa.
            window_steps: liczba kroków czasowych ``T``; ``None`` → wartość
                domyślna z konstruktora.

        Returns:
            Macierz ``np.ndarray`` o kształcie ``(T, P)`` i ``dtype=float32``,
            wypełniona wartościami ``0.0`` / ``1.0``.
        """
        steps = self.window_steps if window_steps is None else int(window_steps)
        if steps <= 0:
            raise ValueError(
                f"window_steps musi być dodatnie, otrzymano: {steps!r}."
            )

        roll = np.zeros((steps, self.n_pitches), dtype=np.float32)
        step_ticks = self._step_ticks(repr_.ticks_per_beat)

        for note in repr_.notes:
            if not (self.pitch_low <= note.pitch < self.pitch_high):
                continue
            pitch_idx = note.pitch - self.pitch_low
            start_step = int(round(note.tick / step_ticks))
            n_steps = max(1, int(round(note.duration_ticks / step_ticks)))
            end_step = start_step + n_steps

            lo = max(0, start_step)
            hi = min(steps, end_step)
            if lo < hi:
                roll[lo:hi, pitch_idx] = 1.0

        return roll

    def to_internal(
        self,
        pianoroll: np.ndarray | Sequence[Sequence[float]],
        template: InternalRepr,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> InternalRepr:
        """Rekonstruuje *Reprezentację_Wewnętrzną* z macierzy pianoroll.

        Ciągłe pasma aktywnych kroków w danej kolumnie (wysokości) są scalane w
        pojedyncze zdarzenia nutowe. Metadane (``ticks_per_beat``, ``meta``,
        ``smf_format``) oraz *velocity* i kanał poszczególnych nut pochodzą z
        ``template`` (Wymaganie 5.7) - macierz pianoroll nie przechowuje tych
        informacji.

        Args:
            pianoroll: macierz ``[T × P]`` (binarna lub ciągła z wyjścia
                sigmoidu). Wartości ``>= threshold`` traktowane są jako aktywne.
            template: *Reprezentacja_Wewnętrzna* dostarczająca tempo, metrum,
                kanały i *velocity* (zwykle *Utwór_Wejściowy*).
            threshold: granica binaryzacji wartości ciągłych (domyślnie ``0.5``).

        Returns:
            Zrekonstruowana :class:`InternalRepr` z deterministycznie
            posortowanymi krotkami zdarzeń.
        """
        matrix = np.asarray(pianoroll)
        if matrix.ndim != 2:
            raise ValueError(
                "pianoroll musi być macierzą 2D [T × P], otrzymano kształt "
                f"{matrix.shape!r}."
            )
        active = matrix >= threshold

        step_ticks = self._step_ticks(template.ticks_per_beat)
        default_velocity, default_channel = self._template_defaults(template)
        pitch_lookup = self._template_lookup(template, step_ticks)

        n_cols = matrix.shape[1]
        notes: list[NoteEvent] = []

        for pitch_idx in range(n_cols):
            pitch = self.pitch_low + pitch_idx
            if not (0 <= pitch <= 127):
                # Kolumna poza prawidłowym zakresem MIDI - pomijamy.
                continue
            for start_step, end_step in _runs(active[:, pitch_idx]):
                tick = int(round(start_step * step_ticks))
                duration_ticks = int(round((end_step - start_step) * step_ticks))
                velocity, channel = self._recover_voice(
                    pitch_lookup.get(pitch),
                    start_step,
                    default_velocity,
                    default_channel,
                )
                notes.append(
                    NoteEvent(
                        tick=tick,
                        channel=channel,
                        pitch=pitch,
                        velocity=velocity,
                        duration_ticks=duration_ticks,
                    )
                )

        return InternalRepr(
            ticks_per_beat=template.ticks_per_beat,
            notes=tuple(sorted(notes, key=event_key)),
            meta=template.meta,
            smf_format=template.smf_format,
        )

    # -- pomocnicze ----------------------------------------------------------

    def _step_ticks(self, ticks_per_beat: int) -> float:
        """Liczba ticków SMF przypadająca na pojedynczy krok pianoroll."""
        if ticks_per_beat <= 0:
            raise ValueError(
                "ticks_per_beat musi być dodatnie, otrzymano: "
                f"{ticks_per_beat!r}."
            )
        return ticks_per_beat / self.steps_per_beat

    def _template_lookup(
        self, template: InternalRepr, step_ticks: float
    ) -> dict[int, list[tuple[int, int, int]]]:
        """Buduje mapę ``pitch → [(start_step, velocity, channel), ...]``.

        Pozwala odtworzyć *velocity* i kanał zrekonstruowanej nuty na podstawie
        najbliższego (czasowo) zdarzenia nutowego *template* o tej samej
        wysokości. Listy są posortowane po ``start_step`` dla deterministycznego
        dopasowania.
        """
        lookup: dict[int, list[tuple[int, int, int]]] = {}
        for note in template.notes:
            start_step = int(round(note.tick / step_ticks))
            lookup.setdefault(note.pitch, []).append(
                (start_step, note.velocity, note.channel)
            )
        for entries in lookup.values():
            entries.sort()
        return lookup

    @staticmethod
    def _recover_voice(
        candidates: list[tuple[int, int, int]] | None,
        start_step: int,
        default_velocity: int,
        default_channel: int,
    ) -> tuple[int, int]:
        """Dobiera *velocity* i kanał dla zrekonstruowanej nuty.

        Wybiera zdarzenie *template* o tej samej wysokości i najbliższym
        ``start_step`` (remisy rozstrzygane na korzyść wcześniejszego kroku).
        W braku kandydatów zwraca wartości zastępcze wyznaczone z *template*.
        """
        if not candidates:
            return default_velocity, default_channel
        best = min(
            candidates,
            key=lambda entry: (abs(entry[0] - start_step), entry[0]),
        )
        return best[1], best[2]

    @staticmethod
    def _template_defaults(template: InternalRepr) -> tuple[int, int]:
        """Wyznacza reprezentatywne *velocity* (mediana) i kanał (dominujący)."""
        velocities = [n.velocity for n in template.notes if n.velocity > 0]
        if velocities:
            default_velocity = int(round(float(np.median(velocities))))
            default_velocity = max(1, min(127, default_velocity))
        else:
            default_velocity = _FALLBACK_VELOCITY

        channels = [n.channel for n in template.notes]
        if channels:
            # Counter.most_common jest stabilny względem kolejności wstawiania;
            # przy remisie wybieramy najmniejszy numer kanału dla determinizmu.
            counts = Counter(channels)
            top = max(counts.values())
            default_channel = min(ch for ch, c in counts.items() if c == top)
        else:
            default_channel = _FALLBACK_CHANNEL

        return default_velocity, default_channel


def _runs(column: np.ndarray) -> list[tuple[int, int]]:
    """Zwraca pasma ciągłych wartości ``True`` jako pary ``(start, end)``.

    ``end`` jest indeksem wyłącznym (pierwszy nieaktywny krok za pasmem), więc
    długość pasma w krokach wynosi ``end - start``.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(column):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(column)))
    return runs
