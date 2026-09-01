"""E1.4 experiment closure: tables, plots, provenance, and Markdown report."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPORT_SCHEMA_VERSION = "e1.4.0"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row}) or [
        "sample_id",
        "composer",
        "title",
        "reason",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for field, value in row.items()
                }
            )
    temporary.replace(path)


def _best_summary(results: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        row
        for row in results.get("summaries", [])
        if row["analysis"] == "all_samples"
        and row["variant"] == "composition_full"
        and row["model"] != "dummy_most_frequent"
    ]
    if not candidates:
        raise ValueError("E1.4 requires composition_full non-dummy results")
    return max(candidates, key=lambda row: row["balanced_accuracy"])


def _plot_confusion(summary: dict[str, Any], path: Path) -> None:
    matrix = np.asarray(summary["confusion_matrix"], dtype=float)
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{int(matrix[row, column])}", ha="center", va="center")
    axis.set_xticks(range(len(summary["labels"])), summary["labels"], rotation=30)
    axis.set_yticks(range(len(summary["labels"])), summary["labels"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"OOF confusion — {summary['model']}")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_folds(results: dict[str, Any], best: dict[str, Any], path: Path) -> None:
    rows = [
        row
        for row in results.get("fold_results", [])
        if row["analysis"] == "all_samples"
        and row["variant"] == "composition_full"
        and row["model"] == best["model"]
    ]
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.scatter(range(1, len(rows) + 1), [row["balanced_accuracy"] for row in rows])
    axis.axhline(1 / 3, color="black", linestyle="--", label="chance = 1/3")
    axis.set_xlabel("Outer fold across repeats")
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("Fold-level E1b scores")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_ablations(results: dict[str, Any], best: dict[str, Any], path: Path) -> None:
    rows = [
        row
        for row in results.get("summaries", [])
        if row["analysis"] == "all_samples" and row["model"] == best["model"]
    ]
    rows.sort(key=lambda row: row["balanced_accuracy"], reverse=True)
    fig, axis = plt.subplots(figsize=(10, max(4, len(rows) * 0.35)))
    axis.barh(
        [row["variant"] for row in reversed(rows)],
        [row["balanced_accuracy"] for row in reversed(rows)],
    )
    axis.axvline(1 / 3, color="black", linestyle="--")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Balanced accuracy")
    axis.set_title(f"Feature variants — {best['model']}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_importance(results: dict[str, Any], best: dict[str, Any], path: Path) -> bool:
    rows = [
        row
        for row in results.get("permutation_importance", {}).get("by_feature", [])
        if row["model"] == best["model"]
    ]
    if not rows:
        return False
    rows = sorted(rows, key=lambda row: row["mean_across_folds"], reverse=True)[:20]
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(
        [row["feature"] for row in reversed(rows)],
        [row["mean_across_folds"] for row in reversed(rows)],
    )
    axis.set_xlabel("Permutation importance (balanced-accuracy decrease)")
    axis.set_title(f"Top held-out feature importances — {best['model']}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _permutation_test(results: dict[str, Any], best: dict[str, Any]) -> dict[str, Any] | None:
    matches = [
        row
        for row in results.get("retrained_group_permutation_tests", [])
        if row["variant"] == best["variant"] and row["model"] == best["model"]
    ]
    return matches[0] if matches else None


def _open_markdown(open_results: dict[str, Any] | None) -> list[str]:
    if open_results is None:
        return [
            "## E1-open",
            "",
            "Analiza E1-open nie została dołączona do tego zamknięcia eksperymentu.",
        ]
    lines = [
        "## E1-open",
        "",
        "E1-open jest analizą zakresu stosowalności i nie wchodzi do głównego kryterium sukcesu E1.",
        "",
        "| Model | AUROC | AUPRC | Unknown recall | Known recall | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in open_results.get("summaries", []):
        if row["scenario"] != "external_unknowns":
            continue
        metrics = row["test_metrics"]
        lines.append(
            f"| {row['model']} | {metrics['auroc_known_vs_unknown']:.3f} | "
            f"{metrics['auprc_known_vs_unknown']:.3f} | "
            f"{metrics['unknown_recall_at_threshold']:.3f} | "
            f"{metrics['known_recall_at_threshold']:.3f} | {metrics['coverage']:.3f} |"
        )
    return lines


def _form_markdown(results: dict[str, Any], best: dict[str, Any]) -> list[str]:
    rows = [
        row
        for row in results.get("error_and_form_analysis", {}).get("by_form", [])
        if row["variant"] == best["variant"] and row["model"] == best["model"]
    ]
    if not rows:
        return ["## Analiza form", "", "Brak metadanych form w wynikach."]
    rows.sort(key=lambda row: row["accuracy"])
    lines = [
        "## Analiza form",
        "",
        "| Forma | Liczba predykcji | Accuracy |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['form']} | {row['prediction_count']} | {row['accuracy']:.3f} |"
        )
    return lines


def write_e1_closure_report(
    e1b_run_dir: Path | str,
    report_dir: Path | str,
    *,
    open_run_dir: Path | str | None = None,
    report_doc: Path | str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Close a completed E1b run and optionally attach completed E1-open results."""
    run_dir = Path(e1b_run_dir)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("status") != "completed":
        raise ValueError("E1.4 requires a completed E1b run")
    results_path = run_dir / "e1b_results.json"
    predictions_path = run_dir / "e1b_predictions.json"
    results = _read_json(results_path)
    predictions = _read_json(predictions_path)
    manifest_path = run_dir / "inputs" / "manifest.json"
    manifest = _read_json(manifest_path)
    open_results = None
    open_inputs = None
    if open_run_dir is not None:
        open_dir = Path(open_run_dir)
        open_manifest = _read_json(open_dir / "run_manifest.json")
        if open_manifest.get("status") != "completed":
            raise ValueError("E1.4 requires a completed E1-open run")
        open_path = open_dir / "e1_open_results.json"
        open_results = _read_json(open_path)
        open_inputs = {
            "run_manifest": _sha256(open_dir / "run_manifest.json"),
            "results": _sha256(open_path),
        }
    destination = Path(report_dir)
    plots = destination / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    best = _best_summary(results)
    permutation = _permutation_test(results, best)
    ci = best["balanced_accuracy_95_ci_clustered"]
    criterion = {
        "ci_lower_above_chance": float(ci[0]) > 1 / 3,
        "retrained_permutation_available": permutation is not None,
        "retrained_permutation_p_below_0_05": (
            float(permutation["p_value"]) < 0.05 if permutation else None
        ),
        "tempo_features_absent": True,
    }
    criterion["complete"] = permutation is not None
    criterion["passed"] = bool(
        criterion["complete"]
        and criterion["ci_lower_above_chance"]
        and criterion["retrained_permutation_p_below_0_05"]
    )
    exclusions = [
        {
            "sample_id": row["sample_id"],
            "composer": row["composer"],
            "title": row["title"],
            "reason": row.get("exclusion_reason"),
        }
        for row in manifest.get("samples", [])
        if row.get("validation_status") == "excluded"
    ]
    predictions_csv = destination / "predictions.csv"
    exclusions_csv = destination / "exclusions.csv"
    _write_csv(predictions_csv, predictions)
    _write_csv(exclusions_csv, exclusions)
    _plot_confusion(best, plots / "confusion_matrix.png")
    _plot_folds(results, best, plots / "fold_scores.png")
    _plot_ablations(results, best, plots / "ablations.png")
    has_importance = _plot_importance(results, best, plots / "feature_importance.png")
    decision = (
        "GO — kryterium sukcesu E1 spełnione; można przejść do E2."
        if criterion["passed"]
        else (
            "NO-GO — kompletne kryterium sukcesu E1 nie zostało spełnione."
            if criterion["complete"]
            else "NIEKOMPLETNE — brakuje raportowanego testu permutacyjnego z retreningiem."
        )
    )
    lines = [
        "# Wyniki eksperymentu E1",
        "",
        f"Wygenerowano: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Decyzja",
        "",
        decision,
        "",
        "## Wynik główny closed-set",
        "",
        f"Najlepszy wcześniej zdefiniowany model pełnego wariantu: `{best['model']}`.",
        f"Balanced accuracy: **{best['balanced_accuracy']:.3f}**, "
        f"95% CI klastrowe: **[{ci[0]:.3f}, {ci[1]:.3f}]**, "
        f"macro-F1: **{best['macro_f1']:.3f}**.",
        "",
        f"Retraining permutation p-value: **{permutation['p_value']:.4f}**."
        if permutation
        else "Retraining permutation p-value: **brak — przebieg nie jest finalny**.",
        "",
        "Wynik dotyczy wyłącznie zamkniętego rozróżniania Bacha, Beethovena i Chopina w ASAP. "
        "Nie jest dowodem uniwersalnej atrybucji autorstwa ani izolacji stylu od epoki i formy.",
        "",
        "## Artefakty",
        "",
        "- `predictions.csv` — wszystkie predykcje OOF.",
        "- `exclusions.csv` — wszystkie wykluczenia wraz z przyczyną.",
        "- `plots/confusion_matrix.png` — macierz pomyłek.",
        "- `plots/fold_scores.png` — wyniki zewnętrznych foldów.",
        "- `plots/ablations.png` — warianty i ablacje grup cech.",
    ]
    if has_importance:
        lines.append("- `plots/feature_importance.png` — held-out permutation importance.")
    lines.extend(
        [
            "",
            "## Reprodukowalność",
            "",
            f"- Commit kodu: `{results.get('provenance', {}).get('code_commit', 'unknown')}`",
            f"- Schemat wyników: `{results.get('results_schema_version')}`",
            f"- Schemat cech: `{results.get('feature_schema_version')}`",
            f"- Seedy: `{results.get('protocol', {}).get('seeds')}`",
            f"- Fingerprint danych: `{manifest.get('dataset_fingerprints')}`",
            "",
            *_form_markdown(results, best),
            "",
            *_open_markdown(open_results),
            "",
            "## Ograniczenia",
            "",
            "Korpus jest skorelowany z epoką i formą, liczba grup jest niewielka, a wiele części "
            "należy do tych samych większych dzieł. E1-open, jeśli dołączony, mierzy odrzucanie "
            "wybranych kompozytorów ASAP i również nie stanowi otwartego testu atrybucji autorstwa.",
            "",
        ]
    )
    report_path = destination / "E1.md"
    _atomic_text(report_path, "\n".join(lines))
    if report_doc is not None:
        _atomic_text(Path(report_doc), "\n".join(lines))
    closure = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "success_criterion": criterion,
        "best_result": best,
        "inputs": {
            "e1b_run_manifest": _sha256(run_manifest_path),
            "e1b_results": _sha256(results_path),
            "e1b_predictions": _sha256(predictions_path),
            "dataset_manifest": _sha256(manifest_path),
            "open": open_inputs,
        },
        "outputs": {
            "report": report_path.name,
            "predictions": predictions_csv.name,
            "exclusions": exclusions_csv.name,
            "plots": sorted(path.name for path in plots.iterdir()),
        },
    }
    _atomic_json(destination / "closure_manifest.json", closure)
    return report_path, closure
