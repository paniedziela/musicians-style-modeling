"""Testy jednostkowe *Ewaluatora_Subiektywnego* (zadanie 11.4, Wymagania 7.1-7.3).

Weryfikują :mod:`musicians_style.evaluation.subjective` **bez** realnego
*SoundFontu* ani sprzętu audio - renderer FluidSynth jest zastępowany
zaślepką (``fake_renderer``):

* :meth:`SubjectiveEvaluator.prepare_listening_set` buduje :class:`ListeningSet`
  z par i fragmentów referencyjnych, sygnalizuje brak minimum 10 par (Wymaganie 7.1),
* :meth:`SubjectiveEvaluator.render_to_audio` zapisuje WAV 44,1 kHz / 16 bit mono
  (Wymaganie 7.2) i przycina próbki chronione do 15 s (Wymaganie 7.3),
* struktury danych :class:`ListeningSet` / :class:`ListeningPair` zachowują
  oczekiwane niezmienniki.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import pytest

from musicians_style.evaluation.subjective import (
    DEFAULT_SAMPLE_RATE,
    MIN_LISTENING_PAIRS,
    MOS_DIMENSIONS,
    PROTECTED_CLIP_SECONDS,
    SIGNIFICANCE_MIN_RESPONDENTS,
    FormSpec,
    ListeningPair,
    ListeningSet,
    Response,
    SubjectiveEvaluator,
)


def _make_fake_renderer(seconds: float, sample_rate: int = DEFAULT_SAMPLE_RATE):
    """Zwraca renderer-zaślepkę generujący sygnał o zadanej długości (mono)."""

    def fake_renderer(midi_path: Path, soundfont, sr: int) -> np.ndarray:
        n = int(seconds * sr)
        t = np.linspace(0.0, seconds, n, endpoint=False)
        return 0.5 * np.sin(2 * np.pi * 440.0 * t)

    return fake_renderer


def _touch_midis(tmp_path: Path, count: int) -> list[tuple[Path, Path]]:
    pairs = []
    for i in range(count):
        inp = tmp_path / f"in_{i}.mid"
        out = tmp_path / f"out_{i}.mid"
        inp.write_bytes(b"MThd")  # zawartość nieistotna - renderer jest zaślepką
        out.write_bytes(b"MThd")
        pairs.append((inp, out))
    return pairs


def _read_wav_meta(path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as wav:
        return (
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getframerate(),
            wav.getnframes(),
        )


# -- prepare_listening_set ---------------------------------------------------


def test_prepare_listening_set_basic(tmp_path) -> None:
    pairs = _touch_midis(tmp_path, 12)
    refs = [tmp_path / "ref0.mid", tmp_path / "ref1.mid"]
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(2.0))
    ls = ev.prepare_listening_set(pairs, refs, n_pairs=10)
    assert isinstance(ls, ListeningSet)
    assert ls.n_pairs == 10
    assert ls.meets_minimum
    assert len(ls.reference_fragments) == 2
    # Indeksy par są kolejne od 0.
    assert [p.index for p in ls.pairs] == list(range(10))
    assert all(isinstance(p, ListeningPair) for p in ls.pairs)


def test_prepare_listening_set_truncates_to_n_pairs(tmp_path) -> None:
    pairs = _touch_midis(tmp_path, 20)
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    ls = ev.prepare_listening_set(pairs, n_pairs=10)
    assert ls.n_pairs == 10
    # Zachowana kolejność: pierwsza para wskazuje na in_0/out_0.
    assert ls.pairs[0].input_path.name == "in_0.mid"


def test_prepare_listening_set_below_minimum_flagged(tmp_path) -> None:
    pairs = _touch_midis(tmp_path, 5)
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    ls = ev.prepare_listening_set(pairs, n_pairs=10)
    assert ls.n_pairs == 5
    assert not ls.meets_minimum  # poniżej 10 par (Wymaganie 7.1)


def test_prepare_listening_set_empty_raises(tmp_path) -> None:
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    with pytest.raises(ValueError):
        ev.prepare_listening_set([], n_pairs=10)


def test_prepare_listening_set_invalid_n_pairs(tmp_path) -> None:
    pairs = _touch_midis(tmp_path, 3)
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    with pytest.raises(ValueError):
        ev.prepare_listening_set(pairs, n_pairs=0)


# -- render_to_audio (format WAV, Wymaganie 7.2) -----------------------------


def test_render_to_audio_writes_wav_44100_16bit_mono(tmp_path) -> None:
    midi = tmp_path / "song.mid"
    midi.write_bytes(b"MThd")
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(3.0), clip_seconds=15)
    wav_path = ev.render_to_audio(midi)
    assert wav_path.exists()
    assert wav_path.suffix == ".wav"
    channels, sampwidth, framerate, nframes = _read_wav_meta(wav_path)
    assert channels == 1
    assert sampwidth == 2  # 16 bit
    assert framerate == DEFAULT_SAMPLE_RATE  # 44,1 kHz
    # 3 sekundy sygnału przy braku przycięcia (clip 15 s).
    assert nframes == 3 * DEFAULT_SAMPLE_RATE


def test_render_to_audio_custom_output_path(tmp_path) -> None:
    midi = tmp_path / "song.mid"
    midi.write_bytes(b"MThd")
    target = tmp_path / "audio" / "rendered.wav"
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    result = ev.render_to_audio(midi, output_path=target)
    assert result == target
    assert target.exists()


# -- limit 15 s dla próbek chronionych (Wymaganie 7.3) -----------------------


def test_render_to_audio_protected_clip_limited_to_15s(tmp_path) -> None:
    midi = tmp_path / "protected.mid"
    midi.write_bytes(b"MThd")
    # Renderer generuje 30 s; limit chroniony musi przyciąć do 15 s.
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(30.0), clip_seconds=60)
    wav_path = ev.render_to_audio(midi, protected=True)
    _, _, framerate, nframes = _read_wav_meta(wav_path)
    assert nframes == PROTECTED_CLIP_SECONDS * framerate


def test_render_to_audio_clip_seconds_truncation(tmp_path) -> None:
    midi = tmp_path / "song.mid"
    midi.write_bytes(b"MThd")
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(10.0))
    wav_path = ev.render_to_audio(midi, clip_seconds=4)
    _, _, framerate, nframes = _read_wav_meta(wav_path)
    assert nframes == 4 * framerate


def test_render_to_audio_shorter_than_limit_kept_full(tmp_path) -> None:
    midi = tmp_path / "song.mid"
    midi.write_bytes(b"MThd")
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(2.0))
    wav_path = ev.render_to_audio(midi, clip_seconds=15)
    _, _, framerate, nframes = _read_wav_meta(wav_path)
    assert nframes == 2 * framerate


# -- render_listening_set ----------------------------------------------------


def test_render_listening_set_renders_all(tmp_path) -> None:
    pairs = _touch_midis(tmp_path, 10)
    refs = [tmp_path / "ref0.mid"]
    refs[0].write_bytes(b"MThd")
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    ls = ev.prepare_listening_set(pairs, refs, n_pairs=10)
    result = ev.render_listening_set(ls, tmp_path / "wavs")
    assert len(result["inputs"]) == 10
    assert len(result["outputs"]) == 10
    assert len(result["references"]) == 1
    for group in result.values():
        for p in group:
            assert p.exists() and p.suffix == ".wav"


# -- walidacja konstruktora --------------------------------------------------


def test_invalid_sample_rate_rejected() -> None:
    with pytest.raises(ValueError):
        SubjectiveEvaluator(sample_rate=0, renderer=_make_fake_renderer(1.0))


def test_invalid_clip_seconds_rejected() -> None:
    with pytest.raises(ValueError):
        SubjectiveEvaluator(clip_seconds=0, renderer=_make_fake_renderer(1.0))


def test_default_renderer_missing_soundfont_raises(tmp_path) -> None:
    # Domyślny renderer (bez wstrzykniętej zaślepki) wymaga istniejącego .sf2;
    # przy jego braku zgłaszany jest FileNotFoundError - bez próby syntezy.
    midi = tmp_path / "song.mid"
    midi.write_bytes(b"MThd")
    ev = SubjectiveEvaluator(soundfont=tmp_path / "missing.sf2")
    with pytest.raises(FileNotFoundError):
        ev.render_to_audio(midi)


# -- struktury danych --------------------------------------------------------


def test_listening_set_minimum_property() -> None:
    pair = ListeningPair(index=0, input_path=Path("a.mid"), output_path=Path("b.mid"))
    small = ListeningSet(pairs=(pair,))
    assert small.n_pairs == 1
    assert not small.meets_minimum
    many = ListeningSet(pairs=tuple(
        ListeningPair(index=i, input_path=Path(f"a{i}.mid"), output_path=Path(f"b{i}.mid"))
        for i in range(MIN_LISTENING_PAIRS)
    ))
    assert many.meets_minimum


# ===========================================================================
# Zadanie 11.5: formularz ankietowy i analiza odpowiedzi (Wymagania 7.4, 7.5)
# ===========================================================================


def _make_listening_set(tmp_path: Path, count: int) -> ListeningSet:
    """Buduje deterministyczny ListeningSet z ``count`` par (bez renderowania)."""
    pairs = _touch_midis(tmp_path, count)
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    return ev.prepare_listening_set(pairs, n_pairs=count)


# -- build_form: MOS ---------------------------------------------------------


def test_build_form_mos_generates_two_questions_per_pair(tmp_path) -> None:
    ls = _make_listening_set(tmp_path, 10)
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    spec = ev.build_form(ls, "MOS")
    assert isinstance(spec, FormSpec)
    assert spec.form_type == "MOS"
    assert spec.scale == (1, 5)
    assert spec.n_pairs == 10
    # Dwa wymiary (similarity, quality) na każdą parę.
    assert spec.n_questions == 10 * len(MOS_DIMENSIONS)
    dims = {q.dimension for q in spec.questions}
    assert dims == set(MOS_DIMENSIONS)
    # Każde pytanie MOS ma skalę 1-5 i brak wyboru ABX.
    for q in spec.questions:
        assert q.scale == (1, 5)
        assert q.choices is None


def test_build_form_mos_is_deterministic(tmp_path) -> None:
    ls = _make_listening_set(tmp_path, 6)
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    spec_a = ev.build_form(ls, "MOS")
    spec_b = ev.build_form(ls, "MOS")
    assert spec_a == spec_b
    # Kolejność pytań rosnąco wg pary, w obrębie pary wg MOS_DIMENSIONS.
    ids = [q.question_id for q in spec_a.questions]
    assert ids[:4] == [
        "pair00_similarity",
        "pair00_quality",
        "pair01_similarity",
        "pair01_quality",
    ]


def test_build_form_mos_serializable(tmp_path) -> None:
    import json

    ls = _make_listening_set(tmp_path, 3)
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    spec = ev.build_form(ls, "MOS")
    payload = json.dumps(spec.to_dict())  # nie może rzucać
    assert '"form_type": "MOS"' in payload


# -- build_form: ABX ---------------------------------------------------------


def test_build_form_abx_generates_one_question_per_pair(tmp_path) -> None:
    ls = _make_listening_set(tmp_path, 10)
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    spec = ev.build_form(ls, "ABX")
    assert spec.form_type == "ABX"
    assert spec.scale is None
    assert spec.n_questions == 10
    for q in spec.questions:
        assert q.choices == ("A", "B")
        assert q.scale is None
        assert "A" in q.stimuli and "B" in q.stimuli


def test_build_form_abx_is_deterministic(tmp_path) -> None:
    ls = _make_listening_set(tmp_path, 5)
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    assert ev.build_form(ls, "ABX") == ev.build_form(ls, "ABX")


def test_build_form_invalid_type_raises(tmp_path) -> None:
    ls = _make_listening_set(tmp_path, 3)
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    with pytest.raises(ValueError):
        ev.build_form(ls, "XYZ")  # type: ignore[arg-type]


def test_build_form_empty_set_raises() -> None:
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    empty = ListeningSet(pairs=())
    with pytest.raises(ValueError):
        ev.build_form(empty, "MOS")


# -- analyze_responses: n < 20 (brak testu istotności) -----------------------


def _mos_responses(n_respondents: int, value: float = 4.0) -> list[Response]:
    """Tworzy po jednej odpowiedzi 'similarity' na respondenta."""
    return [
        Response(
            respondent_id=f"r{i}",
            question_id="pair00_similarity",
            value=value,
            dimension="similarity",
        )
        for i in range(n_respondents)
    ]


def test_analyze_responses_below_threshold_no_significance(tmp_path) -> None:
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    # 19 respondentów - poniżej progu 20 (Wymaganie 7.5).
    responses = [
        Response(
            respondent_id=f"r{i}",
            question_id="pair00_similarity",
            value=3.0 + (i % 3),
            dimension="similarity",
        )
        for i in range(SIGNIFICANCE_MIN_RESPONDENTS - 1)
    ]
    report = ev.analyze_responses(responses)
    assert report.n_respondents == SIGNIFICANCE_MIN_RESPONDENTS - 1
    assert report.significance is None  # brak testu istotności
    assert report.note is not None and "Pominięto test istotności" in report.note
    assert "similarity" in report.metrics
    stats = report.metrics["similarity"]
    assert stats.n == SIGNIFICANCE_MIN_RESPONDENTS - 1
    # Statystyki opisowe są obecne i skończone.
    assert math.isfinite(stats.mean)
    assert math.isfinite(stats.ci_low) and math.isfinite(stats.ci_high)


# -- analyze_responses: n >= 20 (test istotności wykonany) -------------------


def test_analyze_responses_at_threshold_runs_significance() -> None:
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    # Dokładnie 20 respondentów z oceną odbiegającą od środka skali (3.0).
    responses = [
        Response(
            respondent_id=f"r{i}",
            question_id="pair00_similarity",
            value=4.0 + (0.1 if i % 2 == 0 else -0.1),
            dimension="similarity",
        )
        for i in range(SIGNIFICANCE_MIN_RESPONDENTS)
    ]
    report = ev.analyze_responses(responses)
    assert report.n_respondents == SIGNIFICANCE_MIN_RESPONDENTS
    assert report.significance is not None
    assert report.significance.test == "t-test-1samp"
    # Średnia ~4.0 vs popmean 3.0 -> różnica istotna.
    assert report.significance.significant is True
    assert math.isfinite(report.significance.p_value)


def test_analyze_responses_counts_unique_respondents() -> None:
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    # 25 odpowiedzi, ale tylko 10 unikalnych respondentów -> brak testu.
    responses = [
        Response(
            respondent_id=f"r{i % 10}",
            question_id=f"pair{i % 5:02d}_quality",
            value=4.0,
            dimension="quality",
        )
        for i in range(25)
    ]
    report = ev.analyze_responses(responses)
    assert report.n_responses == 25
    assert report.n_respondents == 10
    assert report.significance is None  # 10 < 20


# -- Wymaganie 7.4 KRYTYCZNE: brak przycinania do skali [1, 5] ---------------


def test_analyze_responses_mean_not_clipped_above_scale() -> None:
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    # Oceny celowo poza skalą (np. dane zaszumione) - średnia ~7.0.
    responses = [
        Response(
            respondent_id=f"r{i}",
            question_id="pair00_similarity",
            value=7.0,
            dimension="similarity",
        )
        for i in range(5)
    ]
    report = ev.analyze_responses(responses)
    stats = report.metrics["similarity"]
    # Średnia NIE jest przycinana do [1, 5] (Wymaganie 7.4).
    assert stats.mean == pytest.approx(7.0)
    assert stats.mean > 5.0


def test_analyze_responses_ci_not_clipped_to_scale() -> None:
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    # Oceny symetryczne wokół górnej granicy skali (5.0): połowa 4.0, połowa 6.0.
    # Średnia == 5.0, a przedział ufności jest symetryczny -> ci_high > 5.0,
    # co dowodzi BRAKU przycinania górnej granicy do skali (Wymaganie 7.4).
    values = [4.0, 6.0] * 15  # 30 ocen, średnia 5.0
    responses = [
        Response(
            respondent_id=f"r{i}",
            question_id="pair00_quality",
            value=v,
            dimension="quality",
        )
        for i, v in enumerate(values)
    ]
    report = ev.analyze_responses(responses)
    stats = report.metrics["quality"]
    assert stats.mean == pytest.approx(5.0)
    assert stats.ci_low <= stats.mean <= stats.ci_high
    # Górna granica przedziału ufności wykracza poza skalę [1, 5] - nieprzycięta.
    assert stats.ci_high > 5.0


def test_analyze_responses_negative_values_not_clipped() -> None:
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    responses = [
        Response(
            respondent_id=f"r{i}",
            question_id="pair00_similarity",
            value=-2.0,
            dimension="similarity",
        )
        for i in range(4)
    ]
    report = ev.analyze_responses(responses)
    stats = report.metrics["similarity"]
    # Wartości poniżej 1 NIE są przycinane (Wymaganie 7.4).
    assert stats.mean == pytest.approx(-2.0)
    assert stats.mean < 1.0


# -- analyze_responses: walidacja --------------------------------------------


def test_analyze_responses_empty_raises() -> None:
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    with pytest.raises(ValueError):
        ev.analyze_responses([])


def test_analyze_responses_invalid_confidence_level_raises() -> None:
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    with pytest.raises(ValueError):
        ev.analyze_responses(_mos_responses(3), confidence_level=1.5)


def test_analyze_responses_groups_multiple_dimensions() -> None:
    ev = SubjectiveEvaluator(renderer=_make_fake_renderer(1.0))
    responses = []
    for i in range(SIGNIFICANCE_MIN_RESPONDENTS):
        responses.append(
            Response(f"r{i}", "pair00_similarity", 4.0, dimension="similarity")
        )
        responses.append(
            Response(f"r{i}", "pair00_quality", 3.5, dimension="quality")
        )
    report = ev.analyze_responses(responses)
    assert set(report.metrics.keys()) == {"similarity", "quality"}
    assert report.metrics["similarity"].mean == pytest.approx(4.0)
    assert report.metrics["quality"].mean == pytest.approx(3.5)
