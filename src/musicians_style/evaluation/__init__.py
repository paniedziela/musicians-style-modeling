"""Content and distance measures, with optional reporting loaded on demand."""

from .distance import euclidean, mahalanobis, mahalanobis_from_inverse, prepare_mahalanobis
from .content import content_metrics, semantic_midi_equal


__all__ = [
    "euclidean",
    "mahalanobis",
    "prepare_mahalanobis",
    "mahalanobis_from_inverse",
    "content_metrics",
    "semantic_midi_equal",
    "ObjectiveEvaluator",
    "EvaluationReport",
    "StatTestResult",
    "DescriptiveStats",
    "SeriesStats",
    "Metric",
    "generate_comparison_plots",
    "plot_pitch_class_histogram",
    "plot_interval_histogram",
    "plot_scalar_boxplots",
    "SubjectiveEvaluator",
    "ListeningSet",
    "ListeningPair",
    "AudioRenderer",
    "default_fluidsynth_renderer",
    "FormSpec",
    "FormQuestion",
    "FormType",
    "Response",
    "SubjectiveReport",
    "MetricStats",
    "SignificanceResult",
]


def __getattr__(name: str):
    # Preserve the package API without importing plots and statistical reports
    # when an algorithm only needs a content metric or distance function.
    if name in {"DescriptiveStats", "EvaluationReport", "Metric", "ObjectiveEvaluator",
                "SeriesStats", "StatTestResult"}:
        from . import objective as module
    elif name in {"generate_comparison_plots", "plot_interval_histogram",
                  "plot_pitch_class_histogram", "plot_scalar_boxplots"}:
        from . import plots as module
    elif name in {"AudioRenderer", "FormQuestion", "FormSpec", "FormType", "ListeningPair",
                  "ListeningSet", "MetricStats", "Response", "SignificanceResult",
                  "SubjectiveEvaluator", "SubjectiveReport", "default_fluidsynth_renderer"}:
        from . import subjective as module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value
