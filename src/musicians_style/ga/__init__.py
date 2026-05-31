"""Algorytm_Genetyczny: genom, operatory, funkcja dopasowania (Wymagania 4.x).

Pakiet udostępnia komponenty *Algorytmu_Genetycznego* realizującego ewolucyjny
transfer stylu (sekcja *Algorytm_Genetyczny* w ``design.md``):

* :class:`Genome` - niskowymiarowy genotyp (4 parametry rzeczywiste, Wymaganie 4.1),
* :data:`IDENTITY_GENOME` - transformacja tożsamościowa (Property 5, Wymaganie 11.7),
* :func:`apply_transformation` - czysta funkcja aplikująca genotyp do
  *Reprezentacji_Wewnętrznej* MIDI.
"""

from .transformation import apply_transformation
from .types import IDENTITY_GENOME, Genome

__all__ = [
    "Genome",
    "IDENTITY_GENOME",
    "apply_transformation",
]
