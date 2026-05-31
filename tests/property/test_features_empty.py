# Feature: musicians-style-modeling, Property 4: Wektor neutralny dla pustych plików
"""Test własnościowy *Property 4* - wektor neutralny dla pustych plików.

**Property 4: Wektor neutralny dla pustych plików MIDI** (``design.md``, sekcja
*Correctness Properties*).
**Validates: Requirements 2.7, 11.8.**

Dla dowolnej *Reprezentacji_Wewnętrznej* o zerowej liczbie nut (pliki puste lub
wyłącznie z pauzami, generowane przez ``midi_internal_repr(max_notes=0)``)
*Ekstraktor_Cech* zwraca deterministyczny
:data:`~musicians_style.features.constants.NEUTRAL_FEATURE_VECTOR`, nie zgłaszając
przy tym żadnego wyjątku (Wymagania 2.7, 11.8).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings

from musicians_style.features.constants import NEUTRAL_FEATURE_VECTOR
from musicians_style.features.extractor import FeatureExtractor
from musicians_style.midi.types import InternalRepr
from tests.property.strategies import midi_internal_repr

_EXTRACTOR = FeatureExtractor()


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(repr_=midi_internal_repr(max_notes=0))
def test_empty_repr_returns_neutral_vector(repr_: InternalRepr) -> None:
    """Pusty plik → :data:`NEUTRAL_FEATURE_VECTOR` bez wyjątku (Property 4)."""
    # Walidacja założenia strategii: brak nut (plik pusty / wyłącznie pauzy).
    assert len(repr_.notes) == 0

    # extract nie zgłasza wyjątku i zwraca dokładnie wektor neutralny.
    fv = _EXTRACTOR.extract(repr_)
    assert fv == NEUTRAL_FEATURE_VECTOR
