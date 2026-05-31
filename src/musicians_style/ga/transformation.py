"""Aplikacja genotypu jako transformacji MIDI - :func:`apply_transformation`.

Moduł implementuje *czystą* funkcję :func:`apply_transformation`, która stosuje
genotyp :class:`~musicians_style.ga.types.Genome` do *Reprezentacji_Wewnętrznej*
(:class:`~musicians_style.midi.types.InternalRepr`), realizując transformację
*Utworu_Wejściowego* w *Algorytmie_Genetycznym* (Wymaganie 4.1, sekcja
*Algorytm_Genetyczny* w ``design.md``).

Transformacja składa się z czterech niezależnych operacji na zdarzeniach
nutowych:

* **Transpozycja** - do wysokości (``pitch``) dodawana jest zaokrąglona
  transpozycja ``round(g.transpose_semitones)``. Nuty, które po transpozycji
  wykraczają poza prawidłowy zakres MIDI ``[0, 127]``, są **odrzucane**
  (spójnie z pomocniczą funkcją ``transpose`` z ``design.md``).
* **Gęstość rytmiczna** - czasy rozpoczęcia nut (``tick``) są mnożone przez
  ``g.rhythm_density_factor`` (i zaokrąglane). Czas rozpoczęcia jest utrzymywany
  jako nieujemny, aby pozostać zgodnym ze specyfikacją SMF.
* **Długość nut** - długości (``duration_ticks``) są mnożone przez
  ``g.note_duration_factor`` (i zaokrąglane, z utrzymaniem nieujemności).
* **Dynamika** - do ``velocity`` dodawane jest ``g.velocity_offset`` z saturacją
  do zakresu ``[0, 127]`` (wartość ``0`` zachowuje semantykę *note-off*
  zgodnie z kontraktem :class:`NoteEvent`).

Zdarzenia meta (tempo, metrum, tonacja, zmiana programu) oraz ``ticks_per_beat``
i ``smf_format`` są zachowywane bez zmian - transformacja modyfikuje wyłącznie
materiał nutowy, co pozwala zachować strukturę metryczną *Utworu_Wejściowego*
(Wymagania 5.6, 5.7).

Niezmiennik tożsamości (Property 5, Wymaganie 11.7)
---------------------------------------------------
Dla ``g == IDENTITY_GENOME`` (transpozycja ``0``, mnożniki ``1.0``, przesunięcie
``0.0``) funkcja zwraca *Reprezentację_Wewnętrzną* semantycznie identyczną z
wejściem: ``round(tick * 1.0) == tick``, ``pitch + 0 == pitch``,
``velocity + 0 == velocity``. Dzięki temu cechy wyekstrahowane z wyniku są
bit-identyczne z cechami wejścia (wspiera zadanie 6.5).

Czystość funkcji
----------------
:func:`apply_transformation` nie modyfikuje argumentów (``InternalRepr`` i
``NoteEvent`` są niemutowalne) ani żadnego stanu globalnego, a dla identycznych
wejść zwraca identyczny wynik - jest funkcją czystą i deterministyczną.
"""

from __future__ import annotations

from ..midi.types import InternalRepr, NoteEvent, event_key
from .types import IDENTITY_GENOME, Genome

__all__ = ["apply_transformation", "IDENTITY_GENOME", "Genome"]

# Prawidłowy zakres wartości MIDI dla wysokości i dynamiki.
_MIDI_MIN = 0
_MIDI_MAX = 127


def _clamp(value: int, low: int, high: int) -> int:
    """Ogranicza ``value`` do domkniętego przedziału ``[low, high]`` (saturacja)."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def apply_transformation(x: InternalRepr, g: Genome) -> InternalRepr:
    """Stosuje genotyp ``g`` do *Reprezentacji_Wewnętrznej* ``x`` (Wymaganie 4.1).

    Czysta, deterministyczna funkcja transformacji MIDI. Tworzy nową
    :class:`InternalRepr` z przekształconymi zdarzeniami nutowymi; zdarzenia meta
    oraz parametry globalne (``ticks_per_beat``, ``smf_format``) pozostają
    nienaruszone.

    Kolejność operacji na pojedynczej nucie:

    1. ``pitch' = pitch + round(g.transpose_semitones)`` - nuty z ``pitch'``
       poza ``[0, 127]`` są odrzucane.
    2. ``tick' = max(0, round(tick * g.rhythm_density_factor))``.
    3. ``duration' = max(0, round(duration_ticks * g.note_duration_factor))``.
    4. ``velocity' = clamp(round(velocity + g.velocity_offset), 0, 127)``.

    Wynikowe nuty są sortowane deterministycznie wg
    :func:`~musicians_style.midi.types.event_key`, co zachowuje powtarzalną
    kolejność zdarzeń (niezbędną dla testów własnościowych).

    Args:
        x: wejściowa *Reprezentacja_Wewnętrzna* (*Utwór_Wejściowy*).
        g: genotyp transformacji.

    Returns:
        Nowa :class:`InternalRepr` po zastosowaniu transformacji. Dla
        ``g == IDENTITY_GENOME`` wynik jest semantycznie identyczny z ``x``.
    """
    transpose = round(g.transpose_semitones)
    density = g.rhythm_density_factor
    duration_factor = g.note_duration_factor
    velocity_offset = g.velocity_offset

    transformed: list[NoteEvent] = []
    for note in x.notes:
        new_pitch = note.pitch + transpose
        if not (_MIDI_MIN <= new_pitch <= _MIDI_MAX):
            # Nuta poza prawidłowym zakresem MIDI po transpozycji - odrzucana.
            continue

        new_tick = max(0, round(note.tick * density))
        new_duration = max(0, round(note.duration_ticks * duration_factor))
        new_velocity = _clamp(
            round(note.velocity + velocity_offset), _MIDI_MIN, _MIDI_MAX
        )

        transformed.append(
            NoteEvent(
                tick=new_tick,
                channel=note.channel,
                pitch=new_pitch,
                velocity=new_velocity,
                duration_ticks=new_duration,
            )
        )

    return InternalRepr(
        ticks_per_beat=x.ticks_per_beat,
        notes=tuple(sorted(transformed, key=event_key)),
        meta=x.meta,
        smf_format=x.smf_format,
    )
