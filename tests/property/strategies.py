"""Wspólne strategie generatorów Hypothesis dla testów własnościowych (Wymaganie 11).

Moduł centralizuje strategie kompozycyjne (``@st.composite``) wykorzystywane przez
testy własnościowe z sekcji *Correctness Properties* (``design.md``). Wydzielenie
ich do jednego modułu gwarantuje, że wszystkie testy operują na **identycznych,
przetestowanych** generatorach (np. ``midi_internal_repr``), zamiast powielać
subtelne założenia o poprawności danych w każdym pliku testowym.

Najważniejsze założenia poprawności danych
==========================================

``midi_internal_repr`` generuje *Reprezentacje_Wewnętrzne*, które są
**round-trippowalne** (Property 1) - tzn. ``parse(write(r))`` jest semantycznie
równoważne ``r``. Aby to zagwarantować, generator wymusza niezmienniki wynikające
z semantyki SMF i implementacji :mod:`musicians_style.midi`:

* ``velocity`` należy do ``[1, 127]`` - wartość 0 oznacza *note-off* (komunikat
  ``note_on`` z ``velocity == 0`` jest parsowany jako zwolnienie nuty), więc nuta
  o ``velocity == 0`` zniknęłaby po round-tripie.
* dla danej pary ``(channel, pitch)`` nuty **nie nakładają się** w czasie. Parser
  paruje ``note_on``/``note_off`` w kolejności FIFO względem ``(channel, pitch)``;
  nakładające się nuty o tej samej wysokości i kanale mogłyby po round-tripie
  zamienić się długościami. Generator rozwiązuje to zachłannie: w obrębie grupy
  ``(channel, pitch)`` zachowuje nutę tylko wtedy, gdy jej początek nie wyprzedza
  końca poprzednio zachowanej nuty.
* meta-zdarzenia są deduplikowane po ``(kind, tick)`` (co najwyżej jedno zdarzenie
  danego rodzaju na dany tick) i mają payloady zgodne ze schematem parsera/printera
  (mikrosekundy tempa, metrum o mianowniku będącym potęgą 2, nazwy tonacji
  akceptowane przez ``mido``, ``program_change`` z poprawnym kanałem i programem).

Pozostałe strategie
===================

* :func:`malformed_midi_bytes` - losowe bajty dla Property 13 (niepusty opis błędu
  walidacji MIDI).
* :func:`genome_strategy` - genotyp *Algorytmu_Genetycznego* w zakresach roboczych.
* :func:`feature_vector_strategy` - *Wektor_Cech* o skończonych wartościach.
* :func:`psd_covariance` - dodatnio półokreślona macierz kowariancji (Mahalanobis).
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from musicians_style.features.types import (
    FEATURE_VECTOR_LENGTH,
    INTERVAL_HISTOGRAM_BINS,
    PITCH_CLASS_BINS,
    FeatureVector,
)
from musicians_style.ga.types import Genome
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent, event_key

__all__ = [
    "TICKS_PER_BEAT_CHOICES",
    "KEY_SIGNATURE_CHOICES",
    "note_event_strategy",
    "meta_event_list_strategy",
    "midi_internal_repr",
    "malformed_midi_bytes",
    "genome_strategy",
    "feature_vector_strategy",
    "psd_covariance",
]

#: Dozwolone rozdzielczości czasowe SMF (ticki na ćwierćnutę) - sekcja
#: *Strategie generatorów Hypothesis* w ``design.md``.
TICKS_PER_BEAT_CHOICES = (96, 120, 192, 480)

#: Mianowniki metrum dopuszczalne przez ``mido`` (potęgi 2).
_TIME_SIG_DENOMINATORS = (1, 2, 4, 8, 16)

#: Wartości tempa (mikrosekundy na ćwierćnutę) round-trippowalne bez strat;
#: odpowiadają zakresowi ~40-300 BPM.
_TEMPO_US_CHOICES = (200_000, 300_000, 400_000, 500_000, 600_000, 1_000_000, 1_500_000)

#: Nazwy tonacji akceptowane przez ``mido`` (podzbiór bezpieczny dla round-tripu).
KEY_SIGNATURE_CHOICES = (
    "C",
    "G",
    "D",
    "A",
    "E",
    "F",
    "Bb",
    "Eb",
    "Am",
    "Em",
    "Bm",
    "Dm",
    "Gm",
    "Cm",
)

# Granice czasu generowanych zdarzeń (małe wartości → szybkie testy, małe pliki).
_MAX_TICK = 5_000
_MAX_DURATION = 1_000


@st.composite
def note_event_strategy(draw: st.DrawFn) -> NoteEvent:
    """Generuje pojedyncze, poprawne zdarzenie nutowe (``velocity`` w ``[1, 127]``)."""
    return NoteEvent(
        tick=draw(st.integers(min_value=0, max_value=_MAX_TICK)),
        channel=draw(st.integers(min_value=0, max_value=15)),
        pitch=draw(st.integers(min_value=0, max_value=127)),
        velocity=draw(st.integers(min_value=1, max_value=127)),
        duration_ticks=draw(st.integers(min_value=0, max_value=_MAX_DURATION)),
    )


def _resolve_overlaps(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Usuwa nakładające się nuty o tej samej parze ``(channel, pitch)``.

    Parser paruje ``note_on``/``note_off`` w kolejności FIFO względem
    ``(channel, pitch)``. Aby round-trip był jednoznaczny, w obrębie każdej takiej
    grupy zachowujemy zachłannie tylko nuty, których początek nie wyprzedza końca
    poprzednio zachowanej nuty (``tick >= last_end``). Pozostałe są odrzucane.
    """
    groups: dict[tuple[int, int], list[NoteEvent]] = {}
    for note in notes:
        groups.setdefault((note.channel, note.pitch), []).append(note)

    kept: list[NoteEvent] = []
    for group in groups.values():
        last_end = -1
        for note in sorted(group, key=lambda n: (n.tick, n.duration_ticks)):
            if note.tick >= last_end:
                kept.append(note)
                last_end = note.tick + note.duration_ticks
    return kept


@st.composite
def meta_event_list_strategy(draw: st.DrawFn) -> list[MetaEvent]:
    """Generuje listę meta-zdarzeń o payloadach zgodnych ze schematem parsera.

    Zdarzenia są deduplikowane po ``(kind, tick)`` - co najwyżej jedno zdarzenie
    danego rodzaju na dany znacznik czasu - aby uniknąć niejednoznaczności
    kolejności przy round-tripie.
    """
    events: list[MetaEvent] = []

    if draw(st.booleans()):
        events.append(
            MetaEvent(
                tick=draw(st.integers(min_value=0, max_value=_MAX_TICK)),
                kind="tempo",
                payload={"tempo": draw(st.sampled_from(_TEMPO_US_CHOICES))},
            )
        )
    if draw(st.booleans()):
        events.append(
            MetaEvent(
                tick=draw(st.integers(min_value=0, max_value=_MAX_TICK)),
                kind="time_signature",
                payload={
                    "numerator": draw(st.integers(min_value=1, max_value=16)),
                    "denominator": draw(st.sampled_from(_TIME_SIG_DENOMINATORS)),
                },
            )
        )
    if draw(st.booleans()):
        events.append(
            MetaEvent(
                tick=draw(st.integers(min_value=0, max_value=_MAX_TICK)),
                kind="key_signature",
                payload={"key": draw(st.sampled_from(KEY_SIGNATURE_CHOICES))},
            )
        )
    if draw(st.booleans()):
        events.append(
            MetaEvent(
                tick=draw(st.integers(min_value=0, max_value=_MAX_TICK)),
                kind="program_change",
                payload={
                    "program": draw(st.integers(min_value=0, max_value=127)),
                    "channel": draw(st.integers(min_value=0, max_value=15)),
                },
            )
        )

    # Deduplikacja po (kind, tick) - zachowujemy pierwsze wystąpienie.
    seen: set[tuple[str, int]] = set()
    deduped: list[MetaEvent] = []
    for event in events:
        key = (event.kind, event.tick)
        if key not in seen:
            seen.add(key)
            deduped.append(event)
    return deduped


@st.composite
def midi_internal_repr(draw: st.DrawFn, max_notes: int = 200) -> InternalRepr:
    """Generuje poprawną, round-trippowalną *Reprezentację_Wewnętrzną* (Property 1).

    Args:
        max_notes: górna granica liczby losowanych nut (przed rozwiązaniem
            nakładań). ``0`` wymusza pliki puste (Property 4).

    Returns:
        :class:`InternalRepr` z nutami posortowanymi wg
        :func:`~musicians_style.midi.types.event_key`, o niezmiennikach
        gwarantujących round-trip (zob. docstring modułu).
    """
    ticks_per_beat = draw(st.sampled_from(TICKS_PER_BEAT_CHOICES))
    raw_notes = draw(
        st.lists(note_event_strategy(), min_size=0, max_size=max_notes)
    )
    notes = _resolve_overlaps(raw_notes)
    meta = draw(meta_event_list_strategy())
    smf_format = draw(st.sampled_from((0, 1)))
    return InternalRepr(
        ticks_per_beat=ticks_per_beat,
        notes=tuple(sorted(notes, key=event_key)),
        meta=tuple(meta),
        smf_format=smf_format,
    )


@st.composite
def malformed_midi_bytes(draw: st.DrawFn) -> bytes:
    """Generuje losowe (potencjalnie uszkodzone) bajty dla Property 13."""
    return draw(st.binary(min_size=1, max_size=4096))


@st.composite
def genome_strategy(draw: st.DrawFn) -> Genome:
    """Generuje genotyp *Algorytmu_Genetycznego* w zakresach roboczych (Wymaganie 4.1).

    Zakresy odpowiadają :data:`musicians_style.ga.algorithm.WORKING_RANGES`.
    """
    finite = {"allow_nan": False, "allow_infinity": False}
    return Genome(
        transpose_semitones=draw(
            st.floats(min_value=-12.0, max_value=12.0, **finite)
        ),
        rhythm_density_factor=draw(
            st.floats(min_value=0.5, max_value=2.0, **finite)
        ),
        note_duration_factor=draw(
            st.floats(min_value=0.5, max_value=2.0, **finite)
        ),
        velocity_offset=draw(st.floats(min_value=-32.0, max_value=32.0, **finite)),
    )


def _finite_floats(min_value: float, max_value: float) -> st.SearchStrategy[float]:
    """Strategia skończonych liczb zmiennoprzecinkowych w zadanym zakresie."""
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )


@st.composite
def feature_vector_strategy(draw: st.DrawFn) -> FeatureVector:
    """Generuje *Wektor_Cech* o skończonych wartościach (Property 10).

    Wartości są ograniczone co do wielkości, aby ``as_array()`` nie prowadził do
    przepełnienia w obliczeniach odległości (skończony, nieujemny wynik).
    Histogramy mają poprawne kształty ``(12,)`` i ``(25,)``.
    """
    pitch_hist = np.array(
        draw(
            st.lists(
                _finite_floats(0.0, 1.0),
                min_size=PITCH_CLASS_BINS,
                max_size=PITCH_CLASS_BINS,
            )
        ),
        dtype=np.float64,
    )
    interval_hist = np.array(
        draw(
            st.lists(
                _finite_floats(0.0, 1.0),
                min_size=INTERVAL_HISTOGRAM_BINS,
                max_size=INTERVAL_HISTOGRAM_BINS,
            )
        ),
        dtype=np.float64,
    )
    return FeatureVector(
        tempo_bpm=draw(_finite_floats(1.0, 400.0)),
        key=draw(st.sampled_from(("C major", "A minor", "G major", "D minor"))),
        pitch_class_histogram=pitch_hist,
        interval_histogram=interval_hist,
        note_density_per_s=draw(_finite_floats(0.0, 100.0)),
        mean_note_duration_s=draw(_finite_floats(0.0, 60.0)),
        std_note_duration_s=draw(_finite_floats(0.0, 60.0)),
        rest_ratio=draw(_finite_floats(0.0, 1.0)),
    )


@st.composite
def psd_covariance(draw: st.DrawFn, n: int = FEATURE_VECTOR_LENGTH) -> np.ndarray:
    """Generuje dodatnio półokreśloną macierz kowariancji ``[n × n]`` (Mahalanobis).

    Konstrukcja niskorzędowa ``A · Aᵀ`` (gdzie ``A`` ma kształt ``[n × r]`` o
    małym rzędzie ``r``) gwarantuje symetrię i dodatnią półokreśloność dla
    dowolnej macierzy ``A`` o skończonych wpisach. Najczęściej daje to macierz
    **osobliwą** (rząd ``r < n``), co celowo ćwiczy ścieżkę pseudoodwrotności
    w :func:`~musicians_style.evaluation.distance.mahalanobis`.

    Wybór niskiego rzędu (``r`` w ``[1, 6]``) zamiast pełnej macierzy ``[n × n]``
    jest świadomy: dla ``n = FEATURE_VECTOR_LENGTH (42)`` pełne losowanie wymagałoby
    ``42 × 42 = 1764`` wartości, przekraczając wewnętrzny bufor generacji Hypothesis.
    Losujemy więc jedynie ``n × r`` wartości (≤ 252), a użycie
    :func:`hypothesis.extra.numpy.arrays` jest dodatkowo oszczędne pamięciowo.
    """
    rank = draw(st.integers(min_value=1, max_value=6))
    a = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n, rank),
            elements=st.floats(
                min_value=-3.0,
                max_value=3.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    return a @ a.T
