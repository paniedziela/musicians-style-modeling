"""Post-hoc closure analysis for the frozen E3 full run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..evaluation.content import content_metrics, semantic_midi_equal
from ..midi.parser import MidiParser


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else 0.0,
        "median": float(np.median(array)) if array.size else 0.0,
        "std": float(array.std()) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
    }


def _group_values(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["source_group_id"]), []).append(float(row[field]))
    return np.asarray([np.mean(grouped[key]) for key in sorted(grouped)])


def _bootstrap_ci(values: np.ndarray, seed: int, samples: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(samples)]
    return float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))


def _sign_p(values: np.ndarray, seed: int, permutations: int = 9999) -> float:
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = sum(abs(float(np.mean(values * rng.choice((-1.0, 1.0), len(values))))) >= observed for _ in range(permutations))
    return (exceed + 1) / (permutations + 1)


def _holm(raw: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    result: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * raw[key]))
        result[key] = running
    return result


def _identity(row: Mapping[str, Any]) -> bool:
    genome = row.get("best_genome", {})
    return all(float(genome.get(name, 0)) == 0 for name in (
        "transpose_semitones", "rhythm_strength", "duration_strength", "thin_strength", "double_strength"
    ))


def _measure_outputs(config: Any, run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    samples = {str(item["sample_id"]): item for item in manifest["samples"] if item.get("validation_status") == "accepted"}
    parser = MidiParser()
    measured: list[dict[str, Any]] = []
    for row in rows:
        source = parser.parse(config.dataset_root / str(samples[str(row["source_id"])]["score_path"]))
        parse_error: str | None = None
        try:
            output = parser.parse(run_dir / str(row["output_path"]))
        except Exception as exc:  # artifact boundary; preserve the failure in the report
            output = source
            parse_error = f"{type(exc).__name__}: {exc}"
        identity = _identity(row)
        exact = bool(row.get("constraints", {}).get("roundtrip_valid"))
        semantic = parse_error is None and (exact or (identity and semantic_midi_equal(source, output)))
        raw_violations = row.get("constraints", {}).get("violations", ())
        violations = set(raw_violations if isinstance(raw_violations, list) else ([raw_violations] if raw_violations else []))
        if semantic:
            violations.discard("roundtrip")
        if identity:
            violations.discard("polyphony")
        measured.append({
            "task_id": row["task_id"],
            "source_id": row["source_id"],
            "source_composer": row["source_composer"],
            "target_composer": row["target_composer"],
            "identity": identity,
            "artifact_parseable": parse_error is None,
            "parse_error": parse_error,
            "exact_internal_roundtrip": exact,
            "semantic_roundtrip": semantic,
            "revised_feasible": parse_error is None and semantic and not violations,
            "remaining_violations": sorted(violations),
            "content": content_metrics(source, output),
        })
    return measured


def _paired_e2(rows: Sequence[Mapping[str, Any]], e2_run_dir: Path | str | None, seed: int) -> dict[str, Any] | None:
    path = Path(e2_run_dir) / "results.json" if e2_run_dir else None
    if path is None or not path.is_file():
        return None
    e2_rows = json.loads(path.read_text(encoding="utf-8"))["results"]
    indexed = {(row["fold"], row["source_id"], row["target_composer"]): row for row in e2_rows}
    paired = []
    for row in rows:
        key = (row["fold"], row["source_id"], row["target_composer"])
        if key in indexed:
            paired.append({**row, "paired_difference": float(row["delta_p_target"]) - float(indexed[key]["delta_p_target"])})
    if len(paired) != len(rows):
        return None
    grouped: dict[str, list[float]] = {}
    directions: dict[str, list[Mapping[str, Any]]] = {}
    for row in paired:
        grouped.setdefault(str(row["source_group_id"]), []).append(float(row["paired_difference"]))
        directions.setdefault(f"{row['source_composer']}→{row['target_composer']}", []).append(row)
    group_values = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)])
    raw = {
        direction: _sign_p(_group_values(values, "paired_difference"), seed + 10 + index)
        for index, (direction, values) in enumerate(sorted(directions.items()))
    }
    adjusted = _holm(raw)
    return {
        "pair_count": len(paired),
        "mean": float(np.mean([row["paired_difference"] for row in paired])),
        "ci95_clustered": list(_bootstrap_ci(group_values, seed + 2)),
        "p_value": _sign_p(group_values, seed + 3),
        "directions": {
            direction: {
                "mean": float(np.mean([row["paired_difference"] for row in values])),
                "p_value_holm": adjusted[direction],
            }
            for direction, values in sorted(directions.items())
        },
    }


def build_closure_report(
    config: Any,
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    e2_run_dir: Path | str | None,
) -> tuple[str, dict[str, Any]]:
    measured = _measure_outputs(config, run_dir, rows)
    delta = np.asarray([float(row["delta_p_target"]) for row in rows])
    objective = np.asarray([float(row["style_gain"]) for row in rows])
    group_delta = _group_values(rows, "delta_p_target")
    delta_ci = _bootstrap_ci(group_delta, config.main_seed)
    direction_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        direction_rows.setdefault(
            f"{row['source_composer']}→{row['target_composer']}", []
        ).append(row)
    direction_delta = {}
    for index, (direction, values) in enumerate(sorted(direction_rows.items())):
        raw_values = np.asarray([float(row["delta_p_target"]) for row in values])
        grouped_values = _group_values(values, "delta_p_target")
        direction_delta[direction] = {
            "mean": float(raw_values.mean()),
            "ci95_clustered": list(_bootstrap_ci(grouped_values, config.main_seed + 20 + index)),
            "p_value": _sign_p(grouped_values, config.main_seed + 30 + index),
            "positive": int(np.sum(raw_values > 0)),
            "zero": int(np.sum(raw_values == 0)),
            "negative": int(np.sum(raw_values < 0)),
        }
    paired = _paired_e2(rows, e2_run_dir, config.main_seed)
    metric_names = (
        "length_error", "note_count_ratio", "onset_f1", "melody_trigram_jaccard",
        "chroma_cosine", "max_polyphony_output", "mean_polyphony_output",
    )
    content = {name: _stats([float(item["content"][name]) for item in measured]) for name in metric_names}
    technical = {
        "artifact_parseable": sum(item["artifact_parseable"] for item in measured),
        "exact_internal_roundtrip": sum(item["exact_internal_roundtrip"] for item in measured),
        "semantic_roundtrip": sum(item["semantic_roundtrip"] for item in measured),
        "revised_feasible": sum(item["revised_feasible"] for item in measured),
    }
    pearson = float(np.corrcoef(objective, delta)[0, 1]) if objective.std() and delta.std() else 0.0
    criteria = {
        "delta_vs_identity_ci_positive": delta_ci[0] > 0,
        "paired_e3_minus_e2_ci_positive": bool(paired and paired["ci95_clustered"][0] > 0),
        "all_artifacts_parseable": technical["artifact_parseable"] == len(rows),
        "all_semantic_roundtrips_valid": technical["semantic_roundtrip"] == len(rows),
        "all_lengths_within_5_percent": all(item["content"]["length_error"] <= .05 for item in measured),
        "median_melody_similarity_at_least_095": content["melody_trigram_jaccard"]["median"] >= .95,
        "no_result_worse_than_identity_by_e3_objective": bool(np.all(objective >= 0)),
    }
    passed = all(criteria.values())
    analysis = {
        "schema_version": "e3.closure.1.0",
        "created_at_utc": _now(),
        "task_count": len(rows),
        "delta_p_target": {
            **_stats(delta.tolist()),
            "ci95_clustered": list(delta_ci),
            "p_value": _sign_p(group_delta, config.main_seed + 1),
            "positive": int(np.sum(delta > 0)), "zero": int(np.sum(delta == 0)), "negative": int(np.sum(delta < 0)),
            "directions": direction_delta,
        },
        "objective_alignment": {
            "pearson_style_gain_vs_delta_p_target": pearson,
            "objective_improved_but_external_proxy_worsened": int(np.sum((objective > 0) & (delta < 0))),
            "identity_outputs": int(np.sum(objective == 0)),
        },
        "technical_validity": technical,
        "content": content,
        "paired_e2": paired,
        "success_criteria": criteria,
        "decision": "GO_WITH_LIMITATIONS" if passed else "NO_GO",
        "per_task": measured,
    }

    counts = analysis["delta_p_target"]
    lines = [
        "# E3 — transfer kontrolowany ograniczeniami", "", f"Wygenerowano: {_now()}", "", "## Decyzja", "",
        ("**GO z ograniczeniami.** Zamrożone kryteria E3 są spełnione po zastosowaniu semantycznej kontroli MIDI. "
         "Efekt stylometryczny jest mały i nierówny kierunkowo; nie stanowi dowodu pełnego transferu stylu kompozytora."
         if passed else "**NO-GO według zamrożonych kryteriów.** E3 jest kompletnym wynikiem, ale nie przechodzi wszystkich bramek."),
        "", "## Wynik główny", "",
        f"- Zadania: **{len(rows)}**; średnie Δp_target: **{counts['mean']:.4f}**, 95% CI klastrowane: **[{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]**, p = **{counts['p_value']:.4f}**.",
        f"- Wyniki dodatnie / zerowe / ujemne: **{counts['positive']} / {counts['zero']} / {counts['negative']}**.",
    ]
    if paired:
        lines.extend([
            f"- Sparowana różnica E3−E2: **{paired['mean']:.4f}**, 95% CI **[{paired['ci95_clustered'][0]:.4f}, {paired['ci95_clustered'][1]:.4f}]**, p = **{paired['p_value']:.4f}**.",
            "", "| Kierunek | Średnia E3 | E3−E2 | p Holma dla różnicy |", "|---|---:|---:|---:|",
            *[
                f"| {direction} | {direction_delta[direction]['mean']:.4f} | "
                f"{values['mean']:.4f} | {values['p_value_holm']:.4f} |"
                for direction, values in paired["directions"].items()
            ],
        ])
    else:
        lines.extend([
            "- Brak kompletnego wyniku E2 do porównania sparowanego.",
            "", "| Kierunek | Średnia E3 | 95% CI klastrowane |", "|---|---:|---:|",
            *[
                f"| {direction} | {values['mean']:.4f} | "
                f"[{values['ci95_clustered'][0]:.4f}, {values['ci95_clustered'][1]:.4f}] |"
                for direction, values in direction_delta.items()
            ],
        ])
    lines.extend([
        "", "## Zachowanie treści", "", "| Metryka | Średnia | Mediana |", "|---|---:|---:|",
        f"| Błąd długości | {content['length_error']['mean']:.4f} | {content['length_error']['median']:.4f} |",
        f"| Onset-F1 | {content['onset_f1']['mean']:.4f} | {content['onset_f1']['median']:.4f} |",
        f"| Podobieństwo trigramów melodii | {content['melody_trigram_jaccard']['mean']:.4f} | {content['melody_trigram_jaccard']['median']:.4f} |",
        f"| Chroma cosine | {content['chroma_cosine']['mean']:.4f} | {content['chroma_cosine']['median']:.4f} |",
        f"| Stosunek liczby nut | {content['note_count_ratio']['mean']:.4f} | {content['note_count_ratio']['median']:.4f} |",
        "",
        "Onsety i kontur melodii są zwykle zachowane, ale chroma cosine jest wyraźnie niższe. "
        "E3 nie powinno być więc opisywane jako bezstratne zachowanie materiału wysokościowego.",
        "", "## Poprawność techniczna", "",
        f"- Pliki ponownie parsowalne: **{technical['artifact_parseable']}/{len(rows)}**.",
        f"- Ścisła równość obiektów po round-trip: **{technical['exact_internal_roundtrip']}/{len(rows)}**.",
        f"- Równoważność semantyczna zdarzeń MIDI: **{technical['semantic_roundtrip']}/{len(rows)}**.",
        f"- Feasible po korekcie kontraktu identity: **{technical['revised_feasible']}/{len(rows)}**.",
        "",
        "Ścisła równość może zmienić parowanie nakładających się nut o tym samym kanale i wysokości. "
        "Równoważność semantyczna porównuje obserwowalne zbiory note-on/note-off i metadane.",
        "", "## Zgodność funkcji celu z niezależnym ewaluatorem", "",
        f"Korelacja Pearsona między zyskiem celu E3 a Δp_target wynosi **{pearson:.4f}**. "
        f"W **{analysis['objective_alignment']['objective_improved_but_external_proxy_worsened']}** przypadkach cel E3 poprawił się, lecz niezależne proxy spadło. "
        f"W **{analysis['objective_alignment']['identity_outputs']}** zadaniach algorytm pozostał przy identity. "
        "Jest to główne ograniczenie metody.",
        "", "## Kryteria zamknięcia", "",
        *[f"- {'PASS' if value else 'FAIL'} — `{name}`" for name, value in criteria.items()],
        "", "## Artefakty", "",
        f"- `{run_dir / 'closure_analysis.json'}` — analiza post-hoc i metryki per zadanie.",
        "- `results.json` i `tasks/*/output.mid` — niezmienione wyniki pełnego przebiegu.",
    ])
    return "\n".join(lines) + "\n", analysis
