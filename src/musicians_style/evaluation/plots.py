"""Wykresy porównawcze cech w ewaluacji obiektywnej (zadanie 11.3, Wymaganie 6.6).

Moduł generuje **histogramy** i **wykresy pudełkowe** cech *Utworu_Wejściowego*,
*Utworu_Wyjściowego* oraz *Zbioru_Stylu* w formacie nadającym się do osadzenia w
*Dokumencie_Dyplomowym* (PDF lub EPS, sterowane ``evaluation.plot_format`` -
:class:`~musicians_style.config.EvaluationConfig`).

Backend nieinteraktywny
=======================

Moduł wymusza backend ``Agg`` biblioteki ``matplotlib`` (``matplotlib.use("Agg")``)
**przed** importem ``pyplot``, dzięki czemu generowanie wykresów działa w trybie
headless (serwer CI, brak ekranu) bez okien i bez urządzenia graficznego.

Typy wykresów
=============

* :func:`plot_pitch_class_histogram` - zgrupowany wykres słupkowy histogramu klas
  wysokości (12 klas) dla wejścia, wyjścia i średniej *Zbioru_Stylu*.
* :func:`plot_interval_histogram` - zgrupowany wykres słupkowy histogramu
  interwałów melodycznych (25 kubełków, -12..+12 półtonów).
* :func:`plot_scalar_boxplots` - wykresy pudełkowe cech skalarnych
  (tempo, gęstość nut, długości nut, proporcja pauz) rozkładu *Zbioru_Stylu* z
  naniesionymi punktami *Utworu_Wejściowego* i *Utworu_Wyjściowego*.

Funkcja zbiorcza :func:`generate_comparison_plots` zapisuje komplet wykresów do
wskazanego katalogu i zwraca listę utworzonych ścieżek.

*Zbiór_Stylu* może być przekazany jako sekwencja *Wektorów_Cech*
(:class:`~musicians_style.features.types.FeatureVector`) - wtedy wykresy
pudełkowe odzwierciedlają pełny rozkład - albo jako agregat
:class:`~musicians_style.features.types.AggregatedFeatures` (dostępna wówczas
jest jedynie wartość średnia/odchylenie, więc rozkład pudełkowy degeneruje się do
pojedynczego punktu odniesienia z wąsami ``± std``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

# Backend nieinteraktywny - musi zostać ustawiony przed importem pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import po matplotlib.use)
import numpy as np  # noqa: E402

from musicians_style.features.types import (  # noqa: E402
    INTERVAL_HISTOGRAM_BINS,
    PITCH_CLASS_BINS,
    AggregatedFeatures,
    FeatureVector,
)
from musicians_style.logging import get_logger  # noqa: E402

__all__ = [
    "SUPPORTED_PLOT_FORMATS",
    "SCALAR_FEATURES",
    "plot_pitch_class_histogram",
    "plot_interval_histogram",
    "plot_scalar_boxplots",
    "generate_comparison_plots",
]

#: Formaty zapisu wykresów dopuszczone przez Wymaganie 6.6 (osadzenie w LaTeX).
SUPPORTED_PLOT_FORMATS: tuple[str, ...] = ("pdf", "eps")

#: Nazwy klas wysokości (oś X histogramu klas wysokości).
_PITCH_CLASS_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)

#: Cechy skalarne *Wektora_Cech* prezentowane na wykresach pudełkowych.
#: Krotki ``(atrybut, etykieta osi)``.
SCALAR_FEATURES: tuple[tuple[str, str], ...] = (
    ("tempo_bpm", "Tempo [BPM]"),
    ("note_density_per_s", "Gęstość nut [1/s]"),
    ("mean_note_duration_s", "Śr. długość nuty [s]"),
    ("std_note_duration_s", "Odch. długości nuty [s]"),
    ("rest_ratio", "Proporcja pauz"),
)

_log = get_logger("evaluation_plots")


def _normalize_format(plot_format: str) -> str:
    """Waliduje i normalizuje format zapisu wykresu (Wymaganie 6.6)."""
    fmt = str(plot_format).lower().lstrip(".")
    if fmt not in SUPPORTED_PLOT_FORMATS:
        raise ValueError(
            f"Nieobsługiwany format wykresu: {plot_format!r}; "
            f"dozwolone: {', '.join(SUPPORTED_PLOT_FORMATS)} (Wymaganie 6.6)."
        )
    return fmt


def _style_mean_vector(
    style_features: Sequence[FeatureVector] | AggregatedFeatures,
) -> FeatureVector:
    """Zwraca średni *Wektor_Cech* *Zbioru_Stylu* (z agregatu lub sekwencji)."""
    if isinstance(style_features, AggregatedFeatures):
        return style_features.mean
    seq = list(style_features)
    if not seq:
        raise ValueError(
            "Zbiór_Stylu jest pusty - nie można wygenerować wykresów porównawczych."
        )
    arrays = np.stack([fv.as_array() for fv in seq], axis=0)
    return FeatureVector.from_array(arrays.mean(axis=0), key=seq[0].key)


def _style_scalar_samples(
    style_features: Sequence[FeatureVector] | AggregatedFeatures,
    attr: str,
) -> np.ndarray:
    """Zwraca próbki danej cechy skalarnej *Zbioru_Stylu* do wykresu pudełkowego.

    Dla sekwencji *Wektorów_Cech* zwraca wszystkie wartości cechy (pełny rozkład).
    Dla :class:`AggregatedFeatures` zwraca pojedynczą wartość średnią (rozkład
    nieznany), co daje zdegenerowany "boxplot" w punkcie średniej.
    """
    if isinstance(style_features, AggregatedFeatures):
        return np.array([getattr(style_features.mean, attr)], dtype=np.float64)
    return np.array(
        [float(getattr(fv, attr)) for fv in style_features], dtype=np.float64
    )


def _save_figure(fig: "plt.Figure", path: Path, fmt: str) -> Path:
    """Zapisuje i zamyka figurę, zwracając ścieżkę pliku."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format=fmt, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pitch_class_histogram(
    input_features: FeatureVector,
    output_features: FeatureVector,
    style_features: Sequence[FeatureVector] | AggregatedFeatures,
    path: Path | str,
    *,
    plot_format: str = "pdf",
) -> Path:
    """Rysuje zgrupowany histogram klas wysokości (Wymaganie 6.6).

    Args:
        input_features: *Wektor_Cech* *Utworu_Wejściowego*.
        output_features: *Wektor_Cech* *Utworu_Wyjściowego*.
        style_features: *Zbiór_Stylu* (sekwencja *Wektorów_Cech* lub agregat).
        path: ścieżka pliku wynikowego.
        plot_format: ``"pdf"`` lub ``"eps"``.

    Returns:
        Ścieżka zapisanego wykresu.
    """
    fmt = _normalize_format(plot_format)
    style_mean = _style_mean_vector(style_features)

    x = np.arange(PITCH_CLASS_BINS)
    width = 0.27

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width, input_features.pitch_class_histogram, width, label="Utwór wejściowy")
    ax.bar(x, output_features.pitch_class_histogram, width, label="Utwór wyjściowy")
    ax.bar(x + width, style_mean.pitch_class_histogram, width, label="Zbiór stylu (średnia)")
    ax.set_xticks(x)
    ax.set_xticklabels(_PITCH_CLASS_NAMES)
    ax.set_xlabel("Klasa wysokości")
    ax.set_ylabel("Udział znormalizowany")
    ax.set_title("Histogram klas wysokości")
    ax.legend()

    return _save_figure(fig, Path(path), fmt)


def plot_interval_histogram(
    input_features: FeatureVector,
    output_features: FeatureVector,
    style_features: Sequence[FeatureVector] | AggregatedFeatures,
    path: Path | str,
    *,
    plot_format: str = "pdf",
) -> Path:
    """Rysuje zgrupowany histogram interwałów melodycznych (Wymaganie 6.6).

    Args:
        input_features: *Wektor_Cech* *Utworu_Wejściowego*.
        output_features: *Wektor_Cech* *Utworu_Wyjściowego*.
        style_features: *Zbiór_Stylu* (sekwencja *Wektorów_Cech* lub agregat).
        path: ścieżka pliku wynikowego.
        plot_format: ``"pdf"`` lub ``"eps"``.

    Returns:
        Ścieżka zapisanego wykresu.
    """
    fmt = _normalize_format(plot_format)
    style_mean = _style_mean_vector(style_features)

    half = INTERVAL_HISTOGRAM_BINS // 2
    intervals = np.arange(-half, half + 1)
    x = np.arange(INTERVAL_HISTOGRAM_BINS)
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - width, input_features.interval_histogram, width, label="Utwór wejściowy")
    ax.bar(x, output_features.interval_histogram, width, label="Utwór wyjściowy")
    ax.bar(x + width, style_mean.interval_histogram, width, label="Zbiór stylu (średnia)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in intervals], fontsize=7)
    ax.set_xlabel("Interwał melodyczny [półtony]")
    ax.set_ylabel("Udział znormalizowany")
    ax.set_title("Histogram interwałów melodycznych")
    ax.legend()

    return _save_figure(fig, Path(path), fmt)


def plot_scalar_boxplots(
    input_features: FeatureVector,
    output_features: FeatureVector,
    style_features: Sequence[FeatureVector] | AggregatedFeatures,
    path: Path | str,
    *,
    plot_format: str = "pdf",
) -> Path:
    """Rysuje wykresy pudełkowe cech skalarnych (Wymaganie 6.6).

    Dla każdej cechy skalarnej (:data:`SCALAR_FEATURES`) tworzony jest osobny
    podwykres: pudełko rozkładu *Zbioru_Stylu* oraz naniesione punkty
    *Utworu_Wejściowego* (niebieski) i *Utworu_Wyjściowego* (pomarańczowy).

    Args:
        input_features: *Wektor_Cech* *Utworu_Wejściowego*.
        output_features: *Wektor_Cech* *Utworu_Wyjściowego*.
        style_features: *Zbiór_Stylu* (sekwencja *Wektorów_Cech* lub agregat).
        path: ścieżka pliku wynikowego.
        plot_format: ``"pdf"`` lub ``"eps"``.

    Returns:
        Ścieżka zapisanego wykresu.
    """
    fmt = _normalize_format(plot_format)

    n = len(SCALAR_FEATURES)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 4))
    if n == 1:  # pragma: no cover - SCALAR_FEATURES ma > 1 element
        axes = [axes]

    for ax, (attr, label) in zip(axes, SCALAR_FEATURES):
        samples = _style_scalar_samples(style_features, attr)
        ax.boxplot(samples, positions=[1], widths=0.5, showfliers=False)
        in_val = float(getattr(input_features, attr))
        out_val = float(getattr(output_features, attr))
        ax.scatter([1], [in_val], color="tab:blue", zorder=3, label="Wejście")
        ax.scatter([1], [out_val], color="tab:orange", zorder=3, label="Wyjście")
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.legend(fontsize=7)

    fig.suptitle("Cechy skalarne: Zbiór stylu vs. wejście/wyjście")
    fig.tight_layout()

    return _save_figure(fig, Path(path), fmt)


def generate_comparison_plots(
    input_features: FeatureVector,
    output_features: FeatureVector,
    style_features: Sequence[FeatureVector] | AggregatedFeatures,
    output_dir: Path | str,
    *,
    plot_format: str = "pdf",
    prefix: str = "comparison",
) -> list[Path]:
    """Generuje komplet wykresów porównawczych i zwraca listę ścieżek (Wymaganie 6.6).

    Tworzone pliki (w ``output_dir``):

    * ``{prefix}_pitch_class_histogram.{ext}``,
    * ``{prefix}_interval_histogram.{ext}``,
    * ``{prefix}_scalar_boxplots.{ext}``,

    gdzie ``ext`` to ``pdf`` lub ``eps`` zgodnie z ``plot_format``.

    Args:
        input_features: *Wektor_Cech* *Utworu_Wejściowego*.
        output_features: *Wektor_Cech* *Utworu_Wyjściowego*.
        style_features: *Zbiór_Stylu* (sekwencja *Wektorów_Cech* lub agregat
            :class:`AggregatedFeatures`).
        output_dir: katalog docelowy (tworzony, gdy nie istnieje).
        plot_format: format zapisu (``"pdf"`` lub ``"eps"``), zwykle z
            ``EvaluationConfig.plot_format``.
        prefix: przedrostek nazw plików wynikowych.

    Returns:
        Lista ścieżek utworzonych wykresów (w stałej kolejności).
    """
    fmt = _normalize_format(plot_format)
    out = Path(output_dir)

    paths = [
        plot_pitch_class_histogram(
            input_features,
            output_features,
            style_features,
            out / f"{prefix}_pitch_class_histogram.{fmt}",
            plot_format=fmt,
        ),
        plot_interval_histogram(
            input_features,
            output_features,
            style_features,
            out / f"{prefix}_interval_histogram.{fmt}",
            plot_format=fmt,
        ),
        plot_scalar_boxplots(
            input_features,
            output_features,
            style_features,
            out / f"{prefix}_scalar_boxplots.{fmt}",
            plot_format=fmt,
        ),
    ]

    _log.info(
        "wygenerowano wykresy porównawcze cech",
        output_dir=str(out),
        plot_format=fmt,
        n_plots=len(paths),
    )
    return paths
