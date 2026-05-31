"""Funkcje pomocnicze dla testów własnościowych *Ekstraktora_Cech* (Wymaganie 11).

Moduł udostępnia czyste transformacje *Reprezentacji_Wewnętrznej* używane przez
testy własnościowe, których nie należy mylić ze strategiami generatorów Hypothesis
(``strategies.py``). W szczególności :func:`transpose` realizuje transpozycję
wysokości dźwięków wykorzystywaną w teście *Property 6* (równoważność transpozycji,
zadanie 3.6, Wymaganie 11.3).
"""

from __future__ import annotations

from musicians_style.midi.types import InternalRepr, NoteEvent, event_key

__all__ = ["transpose"]

#: Dolna i górna granica poprawnej wysokości dźwięku MIDI (włącznie).
_MIN_PITCH = 0
_MAX_PITCH = 127


def transpose(repr_: InternalRepr, k: int) -> InternalRepr:
    """Transponuje wszystkie nuty *Reprezentacji_Wewnętrznej* o ``k`` półtonów.

    Do wysokości (``pitch``) każdej nuty dodawane jest ``k``. Nuty, których
    transponowana wysokość wykraczałaby poza dopuszczalny zakres MIDI
    ``[0, 127]``, są **odrzucane** (transpozycja nie może wyprodukować nuty o
    niepoprawnej wysokości). Pozostałe pola nuty (``tick``, ``channel``,
    ``velocity``, ``duration_ticks``) są zachowane bez zmian.

    Metadane utworu (``ticks_per_beat``, ``meta``, ``smf_format``) są zachowane
    w niezmienionej postaci, a nuty wynikowe są ponownie deterministycznie
    sortowane wg :func:`~musicians_style.midi.types.event_key`, dzięki czemu
    zwrócona :class:`InternalRepr` zachowuje te same niezmienniki, co wejściowa.

    Args:
        repr_: wejściowa *Reprezentacja_Wewnętrzna*.
        k: liczba półtonów transpozycji (dodatnia w górę, ujemna w dół).

    Returns:
        Nowa :class:`InternalRepr` z transponowanymi (i ewentualnie odfiltrowanymi)
        nutami; pozostałe pola zachowane.
    """
    transposed = [
        NoteEvent(
            tick=note.tick,
            channel=note.channel,
            pitch=note.pitch + k,
            velocity=note.velocity,
            duration_ticks=note.duration_ticks,
        )
        for note in repr_.notes
        if _MIN_PITCH <= note.pitch + k <= _MAX_PITCH
    ]
    return InternalRepr(
        ticks_per_beat=repr_.ticks_per_beat,
        notes=tuple(sorted(transposed, key=event_key)),
        meta=repr_.meta,
        smf_format=repr_.smf_format,
    )
