# Feature: musicians-style-modeling, Property 2: Idempotencja Ekstraktora_Cech
"""Test własnościowy *Property 2* - idempotencja *Ekstraktora_Cech*.

**Property 2: Idempotencja Ekstraktora_Cech** (``design.md``, sekcja
*Correctness Properties*).
**Validates: Requirements 2.4, 11.2.**

Dla dowolnej poprawnej *Reprezentacji_Wewnętrznej* ``r`` dwukrotne wywołanie
:meth:`~musicians_style.features.extractor.FeatureExtractor.extract` zwraca dwa
**bit-identyczne** *Wektory_Cech*. Weryfikujemy to na dwóch poziomach:

* równość wartościowa :class:`~musicians_style.features.types.FeatureVector`
  (operator ``==`` porównuje pola skalarne dokładnie, a tablice element-wise),
* dokładna identyczność tablic ``as_array()`` (``numpy.array_equal``), co
  potwierdza brak jakiegokolwiek niedeterminizmu liczbowego (Wymaganie 2.4).
"""

from __future__ import annotations

import numpy as np
import pytest
from .profiles import property_settings
from hypothesis import given

from musicians_style.features.extractor import FeatureExtractor
from musicians_style.midi.types import InternalRepr
from tests.property.strategies import midi_internal_repr

_EXTRACTOR = FeatureExtractor()


@pytest.mark.property
@property_settings(max_examples=200, deadline=None)
@given(repr_=midi_internal_repr())
def test_extract_is_idempotent(repr_: InternalRepr) -> None:
    """``extract(r)`` jest bit-identyczne dla dwóch niezależnych wywołań."""
    first = _EXTRACTOR.extract(repr_)
    second = _EXTRACTOR.extract(repr_)

    # Równość wartościowa Wektora_Cech (FeatureVector.__eq__).
    assert first == second

    # Bit-identyczność reprezentacji wektorowej (brak niedeterminizmu liczbowego).
    assert np.array_equal(first.as_array(), second.as_array())
