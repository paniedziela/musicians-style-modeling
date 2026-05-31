"""Testy jednostkowe ewaluatora obiektywnego (``musicians_style.evaluation.objective``).

Zakres (zadanie 11.2, Wymagania 6.1-6.5):

* ``build_report`` - reguła doboru testu statystycznego:
  - ``n < 10`` → brak testu, statystyki opisowe, ``statistical_test = None``
    (Wymaganie 6.4),
  - ``n >= 10`` z różnicami normalnymi → test t-Studenta dla prób zależnych
    (Wymaganie 6.3),
  - ``n >= 10`` z różnicami nienormalnymi → test rang Wilcoxona (Wymaganie 6.3),
  - pominięcie statystyk opisowych gdy ``n >= 10`` (Wymaganie 6.3),
* ``write_report`` - raportowanie wartości p w ``objective_report.json``
  (Wymaganie 6.5),
* ``evaluate`` - obliczanie trzech odległości na rzeczywistych plikach MIDI
  (Wymagania 6.1, 6.2), pomijanie par z niepoprawnym plikiem,
* walidacja wejść (metryka, niespójne długości serii).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from musicians_style.evaluation.objective import (
    DescriptiveStats,
    EvaluationReport,
    ObjectiveEvaluator,
    StatTestResult,
)
from musicians_style.features.types import (
    FEATURE_VECTOR_LENGTH,
    INTERVAL_HISTOGRAM_BINS,
    PITCH_CLASS_BINS,
    AggregatedFeatures,
    FeatureVector,
)
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent


# --------------------------------------------------------------------------- #
# Pomocnicze dane
# --------------------------------------------------------------------------- #
def _feature_vector(**overrides: object) -> FeatureVector:
    pch = np.full(PITCH_CLASS_BINS, 1.0 / PITCH_CLASS_BINS, dtype=np.float64)
    ivl = np.zeros(INTERVAL_HISTOGRAM_BINS, dtype=np.float64)
    ivl[12] = 1.0
    params: dict[str, object] = dict(
        tempo_bpm=120.0,
        key="C major",
        pitch_class_histogram=pch,
        interval_histogram=ivl,
        note_density_per_s=2.5,
        mean_note_duration_s=0.4,
        std_note_duration_s=0.1,
        rest_ratio=0.2,
    )
    params.update(overrides)
    return FeatureVector(**params)  # type: ignore[arg-type]


def _aggregated(mean: FeatureVector | None = None) -> AggregatedFeatures:
    mean_vec = mean if mean is not None else _feature_vector()
    cov = np.eye(FEATURE_VECTOR_LENGTH, dtype=np.float64)
    return AggregatedFeatures(
        mean=mean_vec, median=mean_vec, std=mean_vec, covariance=cov
    )


def _write_midi(path: Path, *, pitch: int, n_notes: int = 8) -> Path:
    """Zapisuje prosty, poprawny plik MIDI o ``n_notes`` nutach od ``pitch``."""
    ticks_per_beat = 480
    notes = tuple(
        NoteEvent(
            tick=i * ticks_per_beat,
            channel=0,
            pitch=pitch + (i % 5),
            velocity=80,
            duration_ticks=ticks_per_beat // 2,
        )
        for i in range(n_notes)
    )
    meta = (
        MetaEvent(tick=0, kind="tempo", payload={"tempo": 500_000}),  # 120 BPM
    )
    repr_ = InternalRepr(
        ticks_per_beat=ticks_per_beat, notes=notes, meta=meta, smf_format=1
    )
    MidiPrettyPrinter().write(repr_, path)
    return path


# --------------------------------------------------------------------------- #
# build_report - reguła doboru testu (n < 10)
# --------------------------------------------------------------------------- #
def test_build_report_small_sample_skips_test_and_reports_descriptive() -> None:
    evaluator = ObjectiveEvaluator()
    before = [5.0, 4.0, 4.5, 3.0, 6.0]  # n = 5 < 10
    after = [2.0, 1.5, 2.5, 1.0, 3.0]
    io = [1.0, 1.1, 0.9, 1.2, 0.8]

    report = evaluator.build_report(before, after, io)

    assert report.n_pairs == 5
    assert report.statistical_test is None  # Wymaganie 6.4
    assert isinstance(report.descriptive_stats, DescriptiveStats)
    # Statystyki opisowe odzwierciedlają dane wejściowe.
    assert report.descriptive_stats.to_style_before.count == 5
    assert report.descriptive_stats.to_style_before.mean == pytest.approx(
        float(np.mean(before))
    )
    assert report.descriptive_stats.to_style_after.median == pytest.approx(
        float(np.median(after))
    )


def test_build_report_below_custom_threshold_skips_test() -> None:
    evaluator = ObjectiveEvaluator(min_pairs_for_test=20)
    n = 15
    before = list(np.linspace(5.0, 6.0, n))
    after = list(np.linspace(2.0, 3.0, n))
    io = list(np.full(n, 1.0))

    report = evaluator.build_report(before, after, io)

    assert report.n_pairs == 15
    assert report.statistical_test is None
    assert report.descriptive_stats is not None


# --------------------------------------------------------------------------- #
# build_report - reguła doboru testu (n >= 10)
# --------------------------------------------------------------------------- #
def test_build_report_normal_differences_uses_ttest() -> None:
    evaluator = ObjectiveEvaluator()
    rng = np.random.default_rng(12345)
    n = 30
    # Różnice ~ N(3, 0.5) → bliskie normalnym → test t-Studenta.
    diffs = rng.normal(loc=3.0, scale=0.5, size=n)
    after = rng.normal(loc=2.0, scale=0.3, size=n)
    before = after + diffs
    io = list(rng.normal(loc=1.0, scale=0.2, size=n))

    report = evaluator.build_report(list(before), list(after), io)

    assert report.statistical_test is not None
    assert report.statistical_test.test == "t-test"  # Wymaganie 6.3
    assert report.descriptive_stats is None  # Wymaganie 6.3 - pominięte
    # Wyraźna różnica przed/po → wynik istotny (p < alpha).
    assert report.statistical_test.significant is True
    assert 0.0 <= report.statistical_test.p_value <= 1.0
    assert report.statistical_test.normality_p_value is not None


def test_build_report_non_normal_differences_uses_wilcoxon() -> None:
    evaluator = ObjectiveEvaluator()
    n = 12
    # Różnice silnie nienormalne: jedna ekstremalna wartość odstająca.
    before = [3.0] * (n - 1) + [100.0]
    after = [2.0] * (n - 1) + [1.0]
    io = [1.0] * n

    report = evaluator.build_report(before, after, io)

    assert report.statistical_test is not None
    assert report.statistical_test.test == "wilcoxon"  # Wymaganie 6.3
    assert report.descriptive_stats is None
    assert 0.0 <= report.statistical_test.p_value <= 1.0


def test_build_report_exactly_ten_pairs_runs_test() -> None:
    evaluator = ObjectiveEvaluator()
    n = 10  # próg dokładnie spełniony
    before = list(np.linspace(5.0, 6.0, n))
    after = list(np.linspace(2.0, 2.5, n))
    io = list(np.full(n, 1.0))

    report = evaluator.build_report(before, after, io)

    assert report.n_pairs == 10
    assert report.statistical_test is not None  # Wymaganie 6.3 (n >= 10)


def test_build_report_identical_before_after_not_significant() -> None:
    evaluator = ObjectiveEvaluator()
    n = 15
    same = list(np.linspace(2.0, 4.0, n))
    io = list(np.full(n, 1.0))

    report = evaluator.build_report(same, list(same), io)

    assert report.statistical_test is not None
    # Brak różnicy → wynik nieistotny statystycznie.
    assert report.statistical_test.significant is False
    assert report.statistical_test.p_value == pytest.approx(1.0)


def test_build_report_mismatched_series_lengths_raise() -> None:
    evaluator = ObjectiveEvaluator()
    with pytest.raises(ValueError, match="identyczną długość"):
        evaluator.build_report([1.0, 2.0], [1.0], [1.0, 2.0])


# --------------------------------------------------------------------------- #
# write_report - raportowanie wartości p (Wymaganie 6.5)
# --------------------------------------------------------------------------- #
def test_write_report_persists_p_value(tmp_path: Path) -> None:
    evaluator = ObjectiveEvaluator()
    n = 20
    before = list(np.linspace(5.0, 7.0, n))
    after = list(np.linspace(2.0, 2.5, n))
    io = list(np.full(n, 1.0))
    report = evaluator.build_report(before, after, io)

    out_path = evaluator.write_report(report, tmp_path / "objective_report.json")

    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["n_pairs"] == 20
    assert data["metric"] == "euclidean"
    assert data["statistical_test"] is not None
    # Wartość p jest raportowana (Wymaganie 6.5).
    assert "p_value" in data["statistical_test"]
    assert data["statistical_test"]["p_value"] is not None
    assert data["descriptive_stats"] is None
    assert len(data["distances"]["to_style_before"]) == 20


def test_write_report_small_sample_contains_descriptive(tmp_path: Path) -> None:
    evaluator = ObjectiveEvaluator()
    report = evaluator.build_report([5.0, 4.0, 3.0], [2.0, 1.0, 2.5], [1.0, 1.0, 1.0])

    out_path = evaluator.write_report(report, tmp_path / "objective_report.json")

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["statistical_test"] is None
    assert data["descriptive_stats"] is not None
    assert data["descriptive_stats"]["to_style_before"]["count"] == 3


# --------------------------------------------------------------------------- #
# evaluate - obliczanie odległości na rzeczywistych plikach MIDI
# --------------------------------------------------------------------------- #
def test_evaluate_computes_three_distance_series(tmp_path: Path) -> None:
    # Para: wejście (wysokie nuty) → wyjście (nuty bliższe stylowi docelowemu).
    style_mean = _feature_vector(tempo_bpm=120.0)
    aggregated = _aggregated(style_mean)

    pairs: list[tuple[Path, Path]] = []
    for i in range(3):
        in_path = _write_midi(tmp_path / f"in_{i}.mid", pitch=60)
        out_path = _write_midi(tmp_path / f"out_{i}.mid", pitch=62)
        pairs.append((in_path, out_path))

    evaluator = ObjectiveEvaluator()
    report = evaluator.evaluate(pairs, aggregated)

    assert report.n_pairs == 3
    assert len(report.distances_to_style_before) == 3
    assert len(report.distances_to_style_after) == 3
    assert len(report.distances_input_output) == 3
    # Wszystkie odległości skończone i nieujemne.
    for series in (
        report.distances_to_style_before,
        report.distances_to_style_after,
        report.distances_input_output,
    ):
        assert all(np.isfinite(series))
        assert all(v >= 0.0 for v in series)
    # n < 10 → brak testu, statystyki opisowe obecne.
    assert report.statistical_test is None
    assert report.descriptive_stats is not None


def test_evaluate_skips_pair_with_invalid_midi(tmp_path: Path) -> None:
    aggregated = _aggregated()

    good_in = _write_midi(tmp_path / "good_in.mid", pitch=60)
    good_out = _write_midi(tmp_path / "good_out.mid", pitch=62)
    broken = tmp_path / "broken.mid"
    broken.write_bytes(b"not a midi file at all")

    evaluator = ObjectiveEvaluator()
    report = evaluator.evaluate([(good_in, good_out), (broken, good_out)], aggregated)

    # Niepoprawna para pominięta - liczona tylko poprawna.
    assert report.n_pairs == 1


def test_evaluate_mahalanobis_metric(tmp_path: Path) -> None:
    aggregated = _aggregated()
    in_path = _write_midi(tmp_path / "in.mid", pitch=60)
    out_path = _write_midi(tmp_path / "out.mid", pitch=64)

    evaluator = ObjectiveEvaluator(metric="mahalanobis")
    report = evaluator.evaluate([(in_path, out_path)], aggregated)

    assert report.n_pairs == 1
    assert np.isfinite(report.distances_to_style_before[0])
    assert report.distances_to_style_before[0] >= 0.0


# --------------------------------------------------------------------------- #
# Walidacja konstruktora
# --------------------------------------------------------------------------- #
def test_invalid_metric_raises() -> None:
    with pytest.raises(ValueError, match="Nieobsługiwana metryka"):
        ObjectiveEvaluator(metric="cosine")  # type: ignore[arg-type]


def test_empty_report_has_no_pairs() -> None:
    evaluator = ObjectiveEvaluator()
    report = evaluator.build_report([], [], [])
    assert report.n_pairs == 0
    assert report.statistical_test is None
    assert report.descriptive_stats is not None
    assert report.descriptive_stats.to_style_before.count == 0


def test_stat_test_result_to_dict_handles_non_finite() -> None:
    # Bezpieczeństwo serializacji: NaN/inf → None (poprawny JSON).
    result = StatTestResult(
        test="t-test",
        statistic=float("inf"),
        p_value=float("nan"),
        alpha=0.05,
        significant=False,
        normality_p_value=None,
    )
    as_dict = result.to_dict()
    assert as_dict["statistic"] is None
    assert as_dict["p_value"] is None


def test_evaluation_report_default_plots_empty() -> None:
    report = EvaluationReport(
        distances_to_style_before=[1.0],
        distances_to_style_after=[2.0],
        distances_input_output=[0.5],
    )
    assert report.plots == []
