# Feature: musicians-style-modeling, Property 6: Równoważność transpozycji
"""Test własnościowy *Property 6* - równoważność transpozycji.

**Property 6: Równoważność transpozycji** (``design.md``, sekcja
*Correctness Properties*).
**Validates: Requirements 11.3.**

Dla dowolnej poprawnej *Reprezentacji_Wewnętrznej* ``r`` oraz dowolnego
``k ∈ [-12, +12]`` histogram klas wysokości pliku transponowanego o ``k``
półtonów jest **cyklicznym przesunięciem** histogramu klas wysokości pliku
oryginalnego o ``k mod 12`` pozycji.

Niezmiennik ten zachodzi jednak **wyłącznie wtedy, gdy żadna nuta nie zostaje
odrzucona** przez transpozycję (tzn. wszystkie transponowane wysokości mieszczą
się w zakresie MIDI ``[0, 127]``). Odrzucenie choćby jednej nuty zmienia liczność
klas wysokości, a tym samym sam histogram - dlatego takie przypadki są
wykluczane z asercji za pomocą :func:`hypothesis.assume`.
"""

from __future__ import annotations

import numpy as np
import pytest
from .profiles import property_settings
from hypothesis import assume, given
from hypothesis import strategies as st

from musicians_style.features.extractor import FeatureExtractor
from musicians_style.features.types import PITCH_CLASS_BINS
from musicians_style.midi.types import InternalRepr
from tests.property.helpers import transpose
from tests.property.strategies import midi_internal_repr

_EXTRACTOR = FeatureExtractor()


@pytest.mark.property
@property_settings(max_examples=200, deadline=None)
@given(
    repr_=midi_internal_repr(),
    k=st.integers(min_value=-12, max_value=12),
)
def test_transpose_cyclically_shifts_pitch_class_histogram(
    repr_: InternalRepr, k: int
) -> None:
    """Histogram klas wysokości transpozycji = cykliczne przesunięcie o ``k`` (Property 6)."""
    transposed = transpose(repr_, k)

    # Niezmiennik zachodzi tylko gdy transpozycja nie odrzuciła żadnej nuty;
    # odrzucenie nuty zmienia histogram, więc takie przypadki pomijamy.
    assume(len(transposed.notes) == len(repr_.notes))

    original_hist = _EXTRACTOR.extract(repr_).pitch_class_histogram
    transposed_hist = _EXTRACTOR.extract(transposed).pitch_class_histogram

    # Oczekiwane: cykliczne przesunięcie histogramu oryginału o k mod 12 pozycji.
    expected = np.roll(np.asarray(original_hist), k % PITCH_CLASS_BINS)

    assert np.allclose(np.asarray(transposed_hist), expected)
