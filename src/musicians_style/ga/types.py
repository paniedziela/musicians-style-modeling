"""Genotyp *Algorytmu_Genetycznego* - klasa :class:`Genome` (Wymaganie 4.1).

Moduł definiuje niskowymiarowy, interpretowalny muzycznie genotyp *Algorytmu_
Genetycznego* (sekcja *Algorytm_Genetyczny* w ``design.md``). Osobnik populacji
koduje transformację *Utworu_Wejściowego* jako wektor czterech parametrów
rzeczywistych:

* ``transpose_semitones`` - transpozycja w półtonach (zaokrąglana do liczby
  całkowitej przy aplikacji),
* ``rhythm_density_factor`` - mnożnik czasów rozpoczęcia nut (gęstość rytmiczna),
* ``note_duration_factor`` - mnożnik długości nut,
* ``velocity_offset`` - przesunięcie dynamiki (*velocity*).

Zgodnie z Wymaganiem 4.1 poszczególne parametry mogą przyjmować wartości
**ujemne**, reprezentujące transformacje odwrotne (np. transpozycja w dół,
redukcja gęstości rytmicznej, skrócenie długości nut, ściszenie).

Genotyp jest niskowymiarowy (4 parametry), dzięki czemu pozostaje
interpretowalny i pozwala zachować strukturę *Utworu_Wejściowego*
(Wymagania 5.6, 5.7).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Genome", "IDENTITY_GENOME"]


@dataclass(frozen=True)
class Genome:
    """Genotyp pojedynczego osobnika *Algorytmu_Genetycznego* (Wymaganie 4.1).

    Niemutowalny (``frozen=True``) zestaw czterech parametrów rzeczywistych
    kodujących transformację MIDI. Niemutowalność zapewnia haszowalność,
    porównywalność po wartości oraz przydatność w testach własnościowych
    (Wymaganie 11) i zachowanie czystości operatorów genetycznych (operatory
    zwracają nowe instancje, nigdy nie modyfikują wejścia).

    Atrybuty:
        transpose_semitones: transpozycja w półtonach. Przy aplikacji
            zaokrąglana do najbliższej liczby całkowitej (``round``); wartości
            ujemne oznaczają transpozycję w dół. Zakres roboczy ``[-12, +12]``.
        rhythm_density_factor: mnożnik czasów rozpoczęcia nut (``tick``)
            sterujący gęstością rytmiczną. ``1.0`` oznacza brak zmian; zakres
            roboczy ``[0.5, 2.0]``.
        note_duration_factor: mnożnik długości nut (``duration_ticks``).
            ``1.0`` oznacza brak zmian; zakres roboczy ``[0.5, 2.0]``.
        velocity_offset: przesunięcie dynamiki dodawane do ``velocity`` każdej
            nuty (z saturacją do ``[0, 127]`` przy aplikacji). ``0.0`` oznacza
            brak zmian; zakres roboczy ``[-32, +32]``; wartości ujemne ściszają.

    Uwaga:
        Klasa przechowuje parametry jako liczby rzeczywiste (``float``) i nie
        wymusza zakresów - dopuszczalne są wartości ujemne (Wymaganie 4.1).
        Ograniczenia zakresów egzekwowane są na poziomie inicjalizacji populacji
        i operatorów mutacji (zadania 6.3, 6.4), a saturacja wartości MIDI
        (wysokość, *velocity*) odbywa się w :func:`~musicians_style.ga.
        transformation.apply_transformation`.
    """

    transpose_semitones: float
    rhythm_density_factor: float
    note_duration_factor: float
    velocity_offset: float


# Transformacja tożsamościowa (Property 5, Wymaganie 11.7): brak transpozycji,
# brak zmiany gęstości i długości nut, brak przesunięcia dynamiki. Aplikacja
# tego genotypu pozostawia *Reprezentację_Wewnętrzną* semantycznie identyczną.
IDENTITY_GENOME = Genome(
    transpose_semitones=0.0,
    rhythm_density_factor=1.0,
    note_duration_factor=1.0,
    velocity_offset=0.0,
)
