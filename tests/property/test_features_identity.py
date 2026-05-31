# Feature: musicians-style-modeling, Property 5
"""Property 5: Niezmienność Ekstraktora_Cech na transformację tożsamościową (zadanie 6.5).

**Validates: Requirements 11.7**

Sekcja *Correctness Properties* (``design.md``):

    *For any* poprawnej *Reprezentacji_Wewnętrznej* ``r``, *Wektor_Cech*
    ``extract(apply_transformation(r, identity_genome))`` jest identyczny z
    ``extract(r)``, gdzie ``identity_genome = Genome(transpose=0,
    rhythm_density_factor=1.0, note_duration_factor=1.0, velocity_offset=0.0)``.

Wymaganie 11.7 (``requirements.md``):

    FOR ALL poprawnych plików MIDI, THE Ekstraktor_Cech SHALL spełniać własność
    niezmienności na transformację tożsamościową: wektor cech pliku poddanego
    transformacji tożsamościowej SHALL być identyczny z wektorem cech pliku
    oryginalnego.

Strategia ``midi_internal_repr`` pochodzi ze wspólnego, przetestowanego modułu
``tests/property/strategies.py`` (nie jest tu redefiniowana). Generuje poprawne,
round-trippowalne *Reprezentacje_Wewnętrzne*. Aplikacja
:data:`~musicians_style.ga.IDENTITY_GENOME` (``round(tick * 1.0) == tick``,
``pitch + 0 == pitch``, ``velocity + 0 == velocity``) pozostawia materiał nutowy
i meta-zdarzenia nienaruszone, więc *Wektor_Cech* musi być **bit-identyczny**.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings

from musicians_style.features.extractor import FeatureExtractor
from musicians_style.ga import IDENTITY_GENOME, apply_transformation
from musicians_style.midi.types import InternalRepr

from .strategies import midi_internal_repr


@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(repr_=midi_internal_repr())
def test_identity_transformation_preserves_feature_vector(
    repr_: InternalRepr,
) -> None:
    """``extract(apply_transformation(r, IDENTITY_GENOME)) == extract(r)`` (Property 5).

    Transformacja tożsamościowa nie zmienia żadnego zdarzenia nutowego ani
    meta-zdarzenia, więc *Ekstraktor_Cech* MUSI zwrócić *Wektor_Cech*
    bit-identyczny z wektorem oryginału (Wymaganie 11.7). Porównanie ``==``
    klasy :class:`~musicians_style.features.types.FeatureVector` jest dokładne
    (pola skalarne ``==``, histogramy element-po-elemencie).
    """
    extractor = FeatureExtractor()

    original = extractor.extract(repr_)
    transformed = extractor.extract(apply_transformation(repr_, IDENTITY_GENOME))

    assert transformed == original, (
        "Transformacja tożsamościowa zmieniła Wektor_Cech "
        "(naruszenie Property 5 / Wymaganie 11.7)."
    )
