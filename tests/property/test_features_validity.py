# Feature: musicians-style-modeling, Property 3: Walidność Wektora_Cech
"""Test własnościowy *Property 3* - walidność *Wektora_Cech*.

**Property 3: Walidność Wektora_Cech** (``design.md``, sekcja
*Correctness Properties*).
**Validates: Requirements 2.1, 2.3.**

Dla dowolnej poprawnej *Reprezentacji_Wewnętrznej* ``r`` *Wektor_Cech*
``fv = extract(r)`` spełnia łącznie niezmienniki walidności:

* ``fv.tempo_bpm`` jest ściśle dodatnie (Wymaganie 2.1),
* ``fv.rest_ratio`` należy do przedziału ``[0.0, 1.0]`` (Wymaganie 2.1),
* suma ``fv.pitch_class_histogram`` wynosi 1 (z tolerancją 1e-9), o ile plik
  zawiera co najmniej jedną nutę; dla pliku pustego zwracany jest *Wektor_Cech*
  neutralny, którego histogram klas wysokości również sumuje się do 1
  (rozkład jednostajny ``1/12``) - niezmiennik sumy ``== 1`` jest więc
  weryfikowany dla obu przypadków,
* długość spłaszczonego *Wektora_Cech* (``len(fv.as_array())``) jest stała
  i równa :data:`~musicians_style.features.types.FEATURE_VECTOR_LENGTH`
  niezależnie od długości wejścia (Wymaganie 2.3).
"""

from __future__ import annotations

import pytest
from .profiles import property_settings
from hypothesis import given

from musicians_style.features.extractor import FeatureExtractor
from musicians_style.features.types import FEATURE_VECTOR_LENGTH
from musicians_style.midi.types import InternalRepr
from tests.property.strategies import midi_internal_repr

_EXTRACTOR = FeatureExtractor()


@pytest.mark.property
@property_settings(max_examples=200, deadline=None)
@given(repr_=midi_internal_repr())
def test_feature_vector_is_valid(repr_: InternalRepr) -> None:
    """*Wektor_Cech* spełnia wszystkie niezmienniki walidności (Property 3)."""
    fv = _EXTRACTOR.extract(repr_)

    # tempo ściśle dodatnie (Wymaganie 2.1).
    assert fv.tempo_bpm > 0.0

    # rest_ratio w przedziale [0, 1] (Wymaganie 2.1).
    assert 0.0 <= fv.rest_ratio <= 1.0

    # Histogram klas wysokości sumuje się do 1 gdy występuje >= 1 nuta.
    # Dla pliku pustego Ekstraktor_Cech zwraca wektor neutralny, którego
    # histogram (rozkład jednostajny 1/12) również sumuje się do 1.
    if len(repr_.notes) >= 1:
        assert abs(float(sum(fv.pitch_class_histogram)) - 1.0) <= 1e-9

    # Stała długość spłaszczonego wektora cech niezależnie od długości wejścia
    # (Wymaganie 2.3).
    assert len(fv.as_array()) == FEATURE_VECTOR_LENGTH
