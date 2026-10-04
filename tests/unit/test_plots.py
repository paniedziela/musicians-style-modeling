"""Testy jednostkowe wykresów porównawczych cech (zadanie 11.3, Wymaganie 6.6).

Weryfikują, że :mod:`musicians_style.evaluation.plots`:

* generuje pliki wykresów (histogramy klas wysokości, interwałów oraz wykresy
  pudełkowe cech skalarnych) w katalogu tymczasowym (``tmp_path``),
* zapisuje je z poprawnym rozszerzeniem zgodnym z ``plot_format`` (PDF lub EPS),
* działa headless (backend ``Agg``) zarówno dla *Zbioru_Stylu* podanego jako
  sekwencja *Wektorów_Cech*, jak i jako :class:`AggregatedFeatures`,
* waliduje nieobsługiwany format.
"""

from __future__ import annotations

import numpy as np
import pytest

from musicians_style.evaluation.plots import (
    SUPPORTED_PLOT_FORMATS,
    generate_comparison_plots,
    plot_interval_histogram,
    plot_pitch_class_histogram,
    plot_scalar_boxplots,
)
from musicians_style.features.types import (
    FEATURE_VECTOR_LENGTH,
    INTERVAL_HISTOGRAM_BINS,
    PITCH_CLASS_BINS,
    AggregatedFeatures,
    FeatureVector,
)

pytestmark = pytest.mark.regression


def _feature_vector(seed: int, tempo: float = 120.0) -> FeatureVector:
    rng = np.random.default_rng(seed)
    pch = rng.random(PITCH_CLASS_BINS)
    pch = pch / pch.sum()
    interval = rng.random(INTERVAL_HISTOGRAM_BINS)
    interval = interval / interval.sum()
    return FeatureVector(
        tempo_bpm=tempo,
        key="C major",
        pitch_class_histogram=pch,
        interval_histogram=interval,
        note_density_per_s=float(rng.uniform(1.0, 8.0)),
        mean_note_duration_s=float(rng.uniform(0.1, 1.0)),
        std_note_duration_s=float(rng.uniform(0.0, 0.5)),
        rest_ratio=float(rng.uniform(0.0, 1.0)),
    )


def _style_sequence(n: int = 12) -> list[FeatureVector]:
    return [_feature_vector(seed=i, tempo=110.0 + i) for i in range(n)]


def _aggregated(seq: list[FeatureVector]) -> AggregatedFeatures:
    arrays = np.stack([fv.as_array() for fv in seq], axis=0)
    cov = np.cov(arrays, rowvar=False)
    return AggregatedFeatures(
        mean=FeatureVector.from_array(arrays.mean(axis=0)),
        median=FeatureVector.from_array(np.median(arrays, axis=0)),
        std=FeatureVector.from_array(arrays.std(axis=0)),
        covariance=cov,
    )


# -- pojedyncze wykresy ------------------------------------------------------


@pytest.mark.parametrize("fmt", SUPPORTED_PLOT_FORMATS)
def test_pitch_class_histogram_created(tmp_path, fmt) -> None:
    inp = _feature_vector(100)
    out = _feature_vector(200)
    style = _style_sequence()
    path = tmp_path / f"pch.{fmt}"
    result = plot_pitch_class_histogram(inp, out, style, path, plot_format=fmt)
    assert result == path
    assert path.exists() and path.stat().st_size > 0
    assert path.suffix == f".{fmt}"


@pytest.mark.parametrize("fmt", SUPPORTED_PLOT_FORMATS)
def test_interval_histogram_created(tmp_path, fmt) -> None:
    inp = _feature_vector(1)
    out = _feature_vector(2)
    style = _style_sequence()
    path = tmp_path / f"interval.{fmt}"
    plot_interval_histogram(inp, out, style, path, plot_format=fmt)
    assert path.exists() and path.stat().st_size > 0


@pytest.mark.parametrize("fmt", SUPPORTED_PLOT_FORMATS)
def test_scalar_boxplots_created(tmp_path, fmt) -> None:
    inp = _feature_vector(1)
    out = _feature_vector(2)
    style = _style_sequence()
    path = tmp_path / f"box.{fmt}"
    plot_scalar_boxplots(inp, out, style, path, plot_format=fmt)
    assert path.exists() and path.stat().st_size > 0


# -- komplet wykresów --------------------------------------------------------


def test_generate_comparison_plots_creates_all_files(tmp_path) -> None:
    inp = _feature_vector(1)
    out = _feature_vector(2)
    style = _style_sequence()
    paths = generate_comparison_plots(inp, out, style, tmp_path, plot_format="pdf")
    assert len(paths) == 3
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
        assert p.suffix == ".pdf"


def test_generate_comparison_plots_eps(tmp_path) -> None:
    inp = _feature_vector(1)
    out = _feature_vector(2)
    style = _style_sequence()
    paths = generate_comparison_plots(inp, out, style, tmp_path, plot_format="eps")
    assert all(p.suffix == ".eps" and p.exists() for p in paths)


def test_generate_comparison_plots_with_aggregated(tmp_path) -> None:
    inp = _feature_vector(1)
    out = _feature_vector(2)
    style = _aggregated(_style_sequence())
    assert style.covariance.shape == (FEATURE_VECTOR_LENGTH, FEATURE_VECTOR_LENGTH)
    paths = generate_comparison_plots(inp, out, style, tmp_path, plot_format="pdf")
    assert len(paths) == 3
    assert all(p.exists() for p in paths)


def test_generate_comparison_plots_creates_missing_dir(tmp_path) -> None:
    inp = _feature_vector(1)
    out = _feature_vector(2)
    style = _style_sequence()
    nested = tmp_path / "plots" / "run1"
    paths = generate_comparison_plots(inp, out, style, nested, plot_format="pdf")
    assert nested.is_dir()
    assert all(p.exists() for p in paths)


# -- walidacja ---------------------------------------------------------------


def test_invalid_format_rejected(tmp_path) -> None:
    inp = _feature_vector(1)
    out = _feature_vector(2)
    style = _style_sequence()
    with pytest.raises(ValueError):
        plot_pitch_class_histogram(
            inp, out, style, tmp_path / "x.png", plot_format="png"
        )


def test_empty_style_sequence_rejected(tmp_path) -> None:
    inp = _feature_vector(1)
    out = _feature_vector(2)
    with pytest.raises(ValueError):
        plot_pitch_class_histogram(inp, out, [], tmp_path / "x.pdf", plot_format="pdf")
