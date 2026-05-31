"""Stałe *Ekstraktora_Cech* - w szczególności *Wektor_Cech* neutralny.

Moduł definiuje :data:`NEUTRAL_FEATURE_VECTOR` - deterministyczny *Wektor_Cech*
zwracany dla plików MIDI pustych lub zawierających wyłącznie pauzy
(Wymagania 2.7 i 11.8; *Property 4*, zadanie 3.5).

Konwencja wektora neutralnego
=============================

Plik pusty / wyłącznie pauzy nie niesie informacji melodyczno-rytmicznej,
dlatego wartości cech dobrano tak, by były **jednoznacznie zdefiniowane,
deterministyczne i muzycznie neutralne**:

- ``tempo_bpm = 120.0`` - tempo domyślne specyfikacji MIDI (spójne z Wymaganiem
  2.2 dla plików bez meta-zdarzenia tempa).
- ``key = "C major"`` - neutralna, kanoniczna tonacja odniesienia (brak
  bemoli/krzyżyków).
- ``pitch_class_histogram`` - rozkład **jednostajny** ``1/12`` w każdej z 12
  klas. Wybór ten spełnia niezmiennik walidności *Wektora_Cech* (suma histogramu
  == 1, Wymaganie 2.3 / *Property 3*), a jednocześnie wyraża brak preferencji
  którejkolwiek klasy wysokości - naturalna interpretacja "braku informacji"
  (maksymalna entropia). Alternatywa w postaci histogramu zerowego naruszałaby
  niezmiennik sumy == 1, dlatego świadomie wybrano rozkład jednostajny.
- ``interval_histogram`` - tablica **zer**. Histogram interwałów nie podlega
  ograniczeniu sumy == 1 (przy zerowej liczbie nut nie występują żadne interwały
  melodyczne), więc zero jest tu poprawną i naturalną wartością.
- ``note_density_per_s = 0.0`` - brak nut.
- ``mean_note_duration_s = 0.0`` oraz ``std_note_duration_s = 0.0`` - brak nut,
  brak długości do uśrednienia.
- ``rest_ratio = 1.0`` - cały materiał to pauza (Wymaganie 2.7 - "wyłącznie
  pauzy"); wartość mieści się w wymaganym przedziale [0, 1].

Stała jest tworzona z tablic ``float64`` tylko do odczytu (gwarantowane przez
``FeatureVector.__post_init__``), więc jest w pełni niemutowalna i może być
bezpiecznie współdzielona między wywołaniami *Ekstraktora_Cech*.
"""

from __future__ import annotations

import numpy as np

from musicians_style.features.types import (
    INTERVAL_HISTOGRAM_BINS,
    PITCH_CLASS_BINS,
    FeatureVector,
)

__all__ = [
    "NEUTRAL_TEMPO_BPM",
    "NEUTRAL_KEY",
    "NEUTRAL_FEATURE_VECTOR",
]

#: Tempo domyślne wektora neutralnego (BPM) - zgodne ze specyfikacją MIDI.
NEUTRAL_TEMPO_BPM = 120.0

#: Neutralna tonacja odniesienia wektora neutralnego.
NEUTRAL_KEY = "C major"

#: Deterministyczny *Wektor_Cech* dla plików pustych / wyłącznie z pauzami
#: (Wymagania 2.7, 11.8). Histogram klas wysokości jest jednostajny (suma == 1),
#: histogram interwałów zerowy, gęstości zerowe, ``rest_ratio == 1.0``.
NEUTRAL_FEATURE_VECTOR = FeatureVector(
    tempo_bpm=NEUTRAL_TEMPO_BPM,
    key=NEUTRAL_KEY,
    pitch_class_histogram=np.full(PITCH_CLASS_BINS, 1.0 / PITCH_CLASS_BINS, dtype=np.float64),
    interval_histogram=np.zeros(INTERVAL_HISTOGRAM_BINS, dtype=np.float64),
    note_density_per_s=0.0,
    mean_note_duration_s=0.0,
    std_note_duration_s=0.0,
    rest_ratio=1.0,
)
