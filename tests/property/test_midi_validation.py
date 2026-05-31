# Feature: musicians-style-modeling, Property 13
"""Property 13: Niepusty opis błędu walidacji MIDI (zadanie 2.5).

**Validates: Requirements 5.5**

Sekcja *Correctness Properties* (``design.md``):

    *For any* sekwencji bajtów ``b`` (poprawnej lub uszkodzonej, w tym losowych
    bajtów spoza specyfikacji SMF), wywołanie ``MidiParser.validate(b)`` albo
    kończy się sukcesem, albo zgłasza wyjątek ``MidiValidationError`` z niepustym
    polem ``message``. Żadne wywołanie nie skutkuje nieobsłużonym wyjątkiem
    (``TypeError``, ``AttributeError``, ``IndexError``) propagującym poza warstwę
    parsera.

Wymaganie 5.5 (``requirements.md``) wymaga, aby walidacja struktury pliku była
wykonywana przed jakimkolwiek przetwarzaniem treści i aby przy wykryciu
uszkodzenia zwracała błąd z niepustym opisem miejsca i typu uszkodzenia - bez
zgłaszania nieobsłużonego wyjątku.

Strategia ``malformed_midi_bytes`` (= ``st.binary(min_size=1, max_size=4096)``)
pochodzi ze wspólnego, przetestowanego modułu ``tests/property/strategies.py``
i celowo generuje dowolne ciągi bajtów - w tym dane całkowicie spoza specyfikacji
SMF - aby zweryfikować odporność warstwy parsera.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings

from musicians_style.errors import MidiValidationError
from musicians_style.midi.parser import MidiParser

from .strategies import malformed_midi_bytes


@pytest.mark.property
@settings(max_examples=300, deadline=None)
@given(data=malformed_midi_bytes())
def test_validate_never_raises_non_domain_exception(data: bytes) -> None:
    """``MidiParser.validate`` zwraca sukces albo ``MidiValidationError`` (Property 13).

    Dla dowolnych bajtów walidacja MUSI albo zakończyć się sukcesem, albo zgłosić
    wyłącznie :class:`MidiValidationError` z niepustym polem ``message``. Każdy
    inny typ wyjątku (``TypeError``, ``AttributeError``, ``IndexError``, ...)
    oznacza naruszenie Wymagania 5.5 i kończy test niepowodzeniem.
    """
    parser = MidiParser()
    try:
        parser.validate(data)
    except MidiValidationError as exc:
        # Niezmiennik Property 13: opis błędu walidacji jest zawsze niepusty.
        assert isinstance(exc.message, str)
        assert exc.message, "MidiValidationError MUST have a non-empty .message"
    except Exception as exc:  # noqa: BLE001 - celowo łapiemy każdy inny typ, by zgłosić błąd
        pytest.fail(
            "MidiParser.validate zgłosił nieobsłużony wyjątek spoza domeny "
            f"(naruszenie Property 13 / Wymaganie 5.5): {exc!r}"
        )


@pytest.mark.property
@settings(max_examples=300, deadline=None)
@given(data=malformed_midi_bytes())
def test_parse_bytes_raises_only_domain_exception(data: bytes) -> None:
    """``MidiParser.parse_bytes`` zgłasza wyłącznie wyjątki domenowe (Property 13).

    Parsowanie surowych bajtów również nie może propagować nieobsłużonych
    wyjątków: dopuszczalne jest jedynie powodzenie lub :class:`MidiValidationError`
    z niepustym opisem (Wymaganie 5.5).
    """
    parser = MidiParser()
    try:
        parser.parse_bytes(data)
    except MidiValidationError as exc:
        assert isinstance(exc.message, str)
        assert exc.message, "MidiValidationError MUST have a non-empty .message"
    except Exception as exc:  # noqa: BLE001 - celowo łapiemy każdy inny typ, by zgłosić błąd
        pytest.fail(
            "MidiParser.parse_bytes zgłosił nieobsłużony wyjątek spoza domeny "
            f"(naruszenie Property 13 / Wymaganie 5.5): {exc!r}"
        )
