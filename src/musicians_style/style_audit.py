"""V2-04 scientific audit. Frozen inputs, separate fitting/scoring, no search."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata

from .asset_paths import resolve_asset_roots
from .content_audit import frozen_files
from .e1.composition_features import FEATURE_SPECS, COMPOSITION_FEATURES_SCHEMA_VERSION, extract_composition_features
from .e3.profile import style_vector
from .midi.parser import MidiParser
from .provenance import collect_provenance, fingerprint, sha256_file, write_json
from .style_metrics import CONTRACT, MEASURES, assert_disjoint, event_profiles, fit_measures, rank_credit, score_measures


def validate_splits(samples: list[dict], splits: dict) -> list[tuple[int, dict]]:
    by_id = {r["sample_id"]: r for r in samples}
    if len(by_id) != len(samples):
        raise ValueError("duplicate canonical sample")
    folds = []
    repeats = set()
    for rep in splits["repetitions"]:
        if rep["repeat"] in repeats:
            raise ValueError("duplicate repeat")
        repeats.add(rep["repeat"])
        seen, fold_ids = [], set()
        for fold in rep["folds"]:
            if fold["fold"] in fold_ids:
                raise ValueError("duplicate fold")
            fold_ids.add(fold["fold"])
            train, test = fold["train"]["sample_ids"], fold["test"]["sample_ids"]
            if set(train) | set(test) != set(by_id):
                raise ValueError("outer coverage")
            assert_disjoint([by_id[s] for s in train], [by_id[s] for s in test])
            inner_seen = []
            for inner in fold["inner_folds"]:
                a, b = inner["train"]["sample_ids"], inner["validation"]["sample_ids"]
                if set(a) | set(b) != set(train):
                    raise ValueError("inner coverage")
                assert_disjoint([by_id[s] for s in a], [by_id[s] for s in b])
                inner_seen.extend(b)
            if sorted(inner_seen) != sorted(train):
                raise ValueError("inner validation repetition")
            seen.extend(test)
            folds.append((rep["repeat"], fold))
        if sorted(seen) != sorted(by_id):
            raise ValueError("outer test repetition")
    return sorted(folds, key=lambda item: (item[0], item[1]["fold"]))


def align_tasks(e2: list[dict], e3: list[dict], samples: list[dict], folds: list[tuple[int, dict]]) -> list[tuple[dict, dict]]:
    """Exact task-ID plus metadata alignment; never rely on list ordering."""
    by_id = {r["sample_id"]: r for r in samples}
    maps = [{r["task_id"]: r for r in rows} for rows in (e2, e3)]
    if any(len(m) != len(rows) for m, rows in zip(maps, (e2, e3))) or set(maps[0]) != set(maps[1]):
        raise ValueError("E2/E3 duplicate or unmatched task IDs")
    composers = {r["composer"] for r in samples}
    expected = {(r["sample_id"], c) for r in samples for c in composers - {r["composer"]}}
    repeat0 = {f["fold"]: f for repeat, f in folds if repeat == 0}
    pairs = []
    for task_id in sorted(maps[0]):
        left, right = (m[task_id] for m in maps)
        for key in ("source_id", "source_group_id", "source_composer", "target_composer", "fold", "seed", "phase"):
            if left[key] != right[key]:
                raise ValueError("E2/E3 task metadata mismatch: " + key)
        source = by_id[left["source_id"]]
        if left["source_group_id"] != source["group_id"] or left["source_composer"] != source["composer"]:
            raise ValueError("task source identity mismatch")
        if left["status"] != "completed" or right["status"] != "completed" or left["phase"] != "full":
            raise ValueError("noncompleted/nonfull frozen task")
        fold = repeat0[left["fold"]]
        if left["source_id"] not in fold["test"]["sample_ids"]:
            raise ValueError("task source is not held out")
        target_train = sorted(s for s in fold["train"]["sample_ids"] if by_id[s]["composer"] == left["target_composer"])
        if sorted(right["profile_train_ids"]) != target_train:
            raise ValueError("historical E3 profile training provenance mismatch")
        training_hashes = {by_id[s]["sha256"] for s in fold["train"]["sample_ids"]}
        if any(r.get("output_sha256") in training_hashes for r in (left, right)):
            raise ValueError("transformed held-out hash overlaps training input")
        pairs.append((left, right))
    actual = Counter((a["source_id"], a["target_composer"]) for a, _ in pairs)
    if set(actual) != expected or any(n != 1 for n in actual.values()):
        raise ValueError("incomplete transfer directions")
    return pairs


def cluster_summary(rows: list[dict], field: str, *, stratified: bool = False) -> dict:
    grouped = defaultdict(list)
    labels = {}
    for row in rows:
        value = row.get(field)
        if value is not None and np.isfinite(value):
            grouped[row["group_id"]].append(float(value))
            labels[row["group_id"]] = row["composer"]
    defined = sum(map(len, grouped.values()))
    result = {"rows": len(rows), "defined": defined, "undefined": len(rows) - defined,
              "groups": len(grouped), "total_groups": len({r["group_id"] for r in rows}),
              "undefined_groups": len({r["group_id"] for r in rows}) - len(grouped), "row_mean": None, "group_mean": None, "ci95": None}
    if not grouped:
        return result
    result["row_mean"] = float(np.mean([v for values in grouped.values() for v in values]))
    strata = defaultdict(list)
    for group in sorted(grouped):
        strata[labels[group] if stratified else "all"].append(float(np.mean(grouped[group])))
    rng = np.random.default_rng(1729)
    estimates = []
    for values in strata.values():
        values = np.asarray(values)
        estimates.append(values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1))
    result.update(group_mean=float(np.mean([np.mean(v) for v in strata.values()])),
                  ci95=np.quantile(np.mean(estimates, axis=0), [0.025, 0.975]).tolist(),
                  groups_by_stratum={k: len(v) for k, v in strata.items()})
    return result


def agreement(rows: list[dict], left: str, right: str) -> dict:
    complete = [r for r in rows if r.get(left) is not None and r.get(right) is not None
                and np.isfinite(r[left]) and np.isfinite(r[right])]
    grouped = defaultdict(list)
    for r in complete:
        grouped[r["group_id"]].append((r[left], r[right]))
    pairs = np.asarray([np.mean(grouped[g], axis=0) for g in sorted(grouped)]).reshape(-1, 2)
    raw = np.asarray([(r[left], r[right]) for r in complete]).reshape(-1, 2)

    def coefficients(values):
        if len(values) < 3 or np.ptp(values[:, 0]) <= 1e-12 or np.ptp(values[:, 1]) <= 1e-12:
            return [None, None]
        return [float(pearsonr(*values.T).statistic), float(spearmanr(*values.T).statistic)]

    result = {"rows": len(rows), "complete_pairs": len(complete), "undefined_pairs": len(rows) - len(complete),
              "groups": len(pairs), "row_coefficients": dict(zip(("pearson", "spearman"), coefficients(raw))),
              "group_coefficients": dict(zip(("pearson", "spearman"), coefficients(pairs))),
              "ci95": None, "undefined_reason": None}
    if any(v is None for v in coefficients(pairs)):
        result["undefined_reason"] = "fewer than 3 work pairs or constant work-mean score"
        return result
    rng = np.random.default_rng(1729)
    resampled = pairs[rng.integers(0, len(pairs), (2000, len(pairs)))]
    def batch_correlation(values):
        centred = values - values.mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(centred[:, :, 0] ** 2, axis=1) * np.sum(centred[:, :, 1] ** 2, axis=1))
        valid = (np.ptp(values[:, :, 0], axis=1) > 1e-12) & (np.ptp(values[:, :, 1], axis=1) > 1e-12)
        answer = np.full(len(values), np.nan)
        answer[valid] = np.sum(centred[valid, :, 0] * centred[valid, :, 1], axis=1) / denominator[valid]
        return answer
    boot = np.column_stack([batch_correlation(resampled), batch_correlation(rankdata(resampled, axis=1))])
    valid = boot[np.isfinite(boot).all(axis=1)]
    result["defined_bootstrap_draws"] = len(valid)
    result["undefined_bootstrap_draws"] = 2000 - len(valid)
    result["ci95"] = dict(zip(("pearson", "spearman"), np.quantile(valid, [0.025, 0.975], axis=0).T.tolist())) if len(valid) else None
    return result


def movement(before: dict, after: dict, source: str, target: str) -> dict:
    def difference(a, b):
        return None if a is None or b is None else float(a - b)
    delta = difference(after[target], before[target])
    margin_before = difference(before[target], before[source])
    margin_after = difference(after[target], after[source])
    credit, tied = rank_credit(after, target)
    return {"input_target": before[target], "output_target": after[target], "input_source": before[source],
            "output_source": after[source], "delta": delta, "source_drop": difference(before[source], after[source]),
            "margin_before": margin_before, "margin_after": margin_after,
            "margin_delta": difference(margin_after, margin_before), "target_credit": credit, "top_composers": tied,
            "negative": None if delta is None else float(delta < -1e-12),
            "null": None if delta is None else float(abs(delta) <= 1e-12)}


def summarize(real: list[dict], outputs: list[dict]) -> dict:
    summary = {"ranking": {}, "real_directions": {}, "movement": {}, "agreement": {}, "degeneracy": {}}
    for m in MEASURES:
        rows = [dict(group_id=r["group_id"], composer=r["composer"], credit=r["ranking"][m]["credit"]) for r in real]
        ranked = cluster_summary(rows, "credit", stratified=True)
        ranked["chance"] = 1 / 3
        ranked["ties"] = sum(len(r["ranking"][m]["top_composers"]) > 1 for r in real)
        ranked["clearly_above_chance"] = ranked["ci95"] is not None and ranked["ci95"][0] > 1 / 3
        summary["ranking"][m] = ranked
        ranked["composer_recalls"] = {c: cluster_summary([r for r in rows if r["composer"] == c], "credit")
                                      for c in sorted({r["composer"] for r in rows})}
        values = [v for r in real + outputs for v in r["scores"][m].values()]
        finite = np.asarray([v for v in values if v is not None and np.isfinite(v)])
        summary["degeneracy"][m] = {"score_slots": len(values), "defined": len(finite), "undefined": len(values) - len(finite),
            "minimum": float(finite.min()) if len(finite) else None, "maximum": float(finite.max()) if len(finite) else None,
            "std": float(finite.std()) if len(finite) else None, "unique_exact": len(set(finite)),
            "populations": {population: {"rows": len(selected), "undefined_score_slots": sum(v is None for r in selected for v in r["scores"][m].values()),
                "exact_unique": len({v for r in selected for v in r["scores"][m].values() if v is not None})}
                for population, selected in {"real": real, **{e: [r for r in outputs if r["experiment"] == e] for e in ("identity", "E2", "E3")}}.items()},
            "real_unique_exact": len({v for r in real for v in r["scores"][m].values() if v is not None}),
            "real_missing_rows": sum(r["ranking"][m]["credit"] is None for r in real)}
        margins = []
        direction_margins = defaultdict(list)
        for r in real:
            for c, value in r["scores"][m].items():
                if c != r["composer"]:
                    true = r["scores"][m][r["composer"]]
                    margin = None if true is None or value is None else true - value
                    item = dict(group_id=r["group_id"], composer=r["composer"], margin=margin,
                                wins=None if margin is None else float(margin > 1e-12))
                    margins.append(item)
                    direction_margins[r["composer"] + "->" + c].append(item)
        ranked["true_vs_other_margin"] = cluster_summary(margins, "margin", stratified=True)
        ranked["true_vs_other_win_rate"] = cluster_summary(margins, "wins", stratified=True)
        for direction, rows in direction_margins.items():
            summary["real_directions"].setdefault(direction, {})[m] = {
                "true_minus_counterexample_margin": cluster_summary(rows, "margin"),
                "true_over_counterexample_rate": cluster_summary(rows, "wins")}
    buckets = defaultdict(list)
    for r in outputs:
        for suffix in ("all", r["composer"] + "->" + r["target_composer"]):
            buckets[r["experiment"] + ":" + suffix].append(r)
    for key, rows in sorted(buckets.items()):
        summary["movement"][key] = {"rows": len(rows), "works": len({r["group_id"] for r in rows}),
            "byte_identity_count": sum(r["byte_identity"] for r in rows),
            "historical_identity_genome_count": sum(r.get("historical_identity_genome", False) for r in rows), "measures": {}}
        flat = []
        for r in rows:
            item = dict(group_id=r["group_id"], composer=r["composer"], historical_objective=r.get("historical_objective"),
                        historical_rf_delta=r.get("historical_rf_delta"), content_on_fraction=r.get("content_on_fraction"))
            item.update({m: r["movement"][m]["delta"] for m in MEASURES})
            flat.append(item)
        for m in MEASURES:
            movement_rows = [dict(group_id=r["group_id"], composer=r["composer"], **r["movement"][m]) for r in rows]
            summary["movement"][key]["measures"][m] = {field: cluster_summary(movement_rows, field) for field in
                ("delta", "source_drop", "margin_delta", "negative", "null", "target_credit")}
            for flag in ("negative", "null"):
                s = summary["movement"][key]["measures"][m][flag]
                s["count"] = sum(r[flag] == 1 for r in movement_rows)
        summary["agreement"][key] = {a + ":" + b: agreement(flat, a, b) for a, b in combinations(MEASURES, 2)}
        for m in MEASURES:
            for other in ("historical_objective", "historical_rf_delta", "content_on_fraction"):
                summary["agreement"][key][m + ":" + other] = agreement(flat, m, other)
    # True-composer affinity avoids averaging three classifier probabilities to a constant.
    flat_real = [dict(group_id=r["group_id"], **{m: r["scores"][m][r["composer"]] for m in MEASURES}) for r in real]
    summary["agreement"]["real_affinity"] = {a + ":" + b: agreement(flat_real, a, b) for a, b in combinations(MEASURES, 2)}
    for c in sorted(real[0]["scores"]["rms67"]) if real else []:
        flat = [dict(group_id=r["group_id"], **{m: r["scores"][m][c] for m in MEASURES}) for r in real]
        summary["agreement"]["real_affinity:" + c] = {a + ":" + b: agreement(flat, a, b) for a, b in combinations(MEASURES, 2)}
    # Paired E3 minus E2 movement, 300 task pairs per measure.
    lookup = {(r["experiment"], r["task_id"]): r for r in outputs}
    paired = []
    for (experiment, task), left in lookup.items():
        if experiment == "E2":
            right = lookup[("E3", task)]
            paired.append(dict(group_id=left["group_id"], composer=left["composer"],
                **{m: None if left["movement"][m]["delta"] is None or right["movement"][m]["delta"] is None else
                   right["movement"][m]["delta"] - left["movement"][m]["delta"] for m in MEASURES}))
    summary["paired_E3_minus_E2"] = {m: cluster_summary(paired, m) for m in MEASURES}
    summary["review_evidence"] = {}
    for m in MEASURES:
        direction_means = {experiment: {key.split(":", 1)[1]: bucket["measures"][m]["delta"]["group_mean"]
            for key, bucket in summary["movement"].items() if key.startswith(experiment + ":") and not key.endswith(":all")}
            for experiment in ("E2", "E3")}
        summary["review_evidence"][m] = {"ranking_above_chance_with_ci": summary["ranking"][m]["clearly_above_chance"],
            "nondegenerate_real_scores": summary["degeneracy"][m]["real_unique_exact"] > 1,
            "score_missingness": summary["degeneracy"][m]["undefined"], "direction_means": direction_means,
            "positive_mean_directions": {e: sum(v is not None and v > 1e-12 for v in d.values()) for e, d in direction_means.items()},
            "E3_negative_count": summary["movement"]["E3:all"]["measures"][m]["negative"]["count"],
            "E3_null_count": summary["movement"]["E3:all"]["measures"][m]["null"]["count"],
            "promotion": "unselected; multi-evidence review required; retain candidate even if ranking fails"}
    return summary


def report(summary: dict, audit: dict) -> str:
    def number(value):
        return "undefined" if value is None else f"{value:.6f}"
    def interval(value):
        return "undefined" if value is None else "[" + ", ".join(number(v) for v in value) + "]"
    lines = ["# V2-04 E1d style-measure audit", "", f"Integrity/completion: {audit['passed']}; runtime {audit['elapsed_seconds']:.2f} s.", "",
        "Higher affinity is closer; movement is output minus source. Each named score stands alone. No objective has been selected.", "",
        "25 outer training fits per family; real works have five held-out observations each (750 pieces, 87 works). Frozen outputs use only their matching repeat-0 fit. Identities use source-self affinity for each of the 300 transfer tasks.", "",
        "## Held-out real-work ranking", "", "Fractional max-score tie credit, macro composer mean of within-work means; 2,000 stratified work bootstrap draws, seed 1729. Chance = 1/3. Intervals are descriptive and conditional on fitted folds.", "",
        "| Measure | Defined / total | Works | Work balanced ranking [95% CI] | True-minus-other margin | Ties | Evidence above chance |",
        "|---|---:|---:|---|---|---:|---|"]
    for m, r in summary["ranking"].items():
        lines.append(f"| {m} | {r['defined']}/{r['rows']} | {r['groups']} | {number(r['group_mean'])} {interval(r['ci95'])} | {r['true_vs_other_margin']['group_mean']:.6f} | {r['ties']} | {r['clearly_above_chance']} |")
    lines += ["", "Real true-minus-counterexample margins by direction are in summary.json, along with composer recalls. Direction means below measure closeness to transfer target; they have their own scale for each measure."]
    lines += ["", "## Movement and null/negative rates", "", "Each direction and pooled result is an equal-work mean. Denominators count transfer tasks, with repeated targets inside source work; identities are 300 task-references from 150 sources, not 300 distinct real pieces.", "",
        "| Experiment / direction | Measure | Defined / tasks | Works | Target movement [95% CI] | Negative count / defined | Null count / defined |",
        "|---|---|---:|---:|---|---|---|"]
    for key, bucket in summary["movement"].items():
        for m, r in bucket["measures"].items():
            d = r["delta"]
            lines.append(f"| {key} | {m} | {d['defined']}/{d['rows']} | {d['groups']} | {number(d['group_mean'])} {interval(d['ci95'])} | {r['negative']['count']}/{r['negative']['defined']} | {r['null']['count']}/{r['null']['defined']} |")
    lines += ["", "## Agreement", "", "Primary coefficients correlate work-mean paired movements; brackets are work bootstrap intervals. Output-level coefficients, all directions, real affinity agreement and undefined cases are in summary.json. Historical E3 gain belongs only to E3; E2/identity comparisons are explicitly undefined.", "",
        "| Population | Measures | Complete rows / total | Works | Pearson [95% CI] | Spearman [95% CI] |", "|---|---|---:|---:|---|---|"]
    for key in ("E2:all", "E3:all", "real_affinity"):
        for name, r in summary["agreement"][key].items():
            ci = r["ci95"] or {}
            lines.append(f"| {key} | {name} | {r['complete_pairs']}/{r['rows']} | {r['groups']} | {number(r['group_coefficients']['pearson'])} {interval(ci.get('pearson'))} | {number(r['group_coefficients']['spearman'])} {interval(ci.get('spearman'))} |")
    lines += ["", "Real_affinity correlations use the real piece's true-composer affinity; candidate-composer-specific coefficients are recorded separately in summary.json."]
    lines += ["", "## Degeneracy and review evidence", "", "| Measure | Defined / score slots | Range | SD | Exact unique | Ranking missing rows |", "|---|---:|---|---:|---:|---:|"]
    for m, r in summary["degeneracy"].items():
        lines.append(f"| {m} | {r['defined']}/{r['score_slots']} | {r['minimum']} .. {r['maximum']} | {r['std']} | {r['unique_exact']} | {r['real_missing_rows']} |")
    lines += ["", "| Measure | Ranking CI above chance | Nondegenerate real scores | Missing slots | E2 / E3 positive mean directions | E3 negatives / nulls | Review disposition |",
              "|---|---|---|---:|---|---|---|"]
    for m, r in summary["review_evidence"].items():
        lines.append(f"| {m} | {r['ranking_above_chance_with_ci']} | {r['nondegenerate_real_scores']} | {r['score_missingness']} | {r['positive_mean_directions']} | {r['E3_negative_count']} / {r['E3_null_count']} | {r['promotion']} |")
    if "event_diagnostics" in summary:
        lines += ["", "Event-profile diagnostics (piece counts, overflow counts and empty profiles):", "", "```json", json.dumps(summary["event_diagnostics"], indent=2), "```", "",
                  "Fit warnings: " + str(summary["fit_diagnostics"]["warnings"]) + ". Exact serialized-score determinism: " + str(summary["fit_diagnostics"]["serialized_score_determinism"]) + "."]
    lines += ["", "Above-chance ranking is strong evidence, not a promotion decision. Review direction signs, null/negative rates, score coverage/spread, disagreements, leakage checks and interpretability together; all failed candidates remain recorded. V2-05 requires another review.", "",
        "## Limitations and provenance", "",
        "RMS67 and Gaussian67 share the frozen 67-component representation and equal family weights. RMS is exactly the negative frozen E3 distance, retaining std<1e-6 ->1.0. Gaussian is the recorded V2 thesis adaptation: exp(-z^2/2), with training-only std=max(target std, 0.05*pooled training std, 1e-6). It is bounded marginal satisfaction rather than likelihood and does not reproduce E3 variance handling. Correlated bins and scale make representation/variance handling material.", "",
        "Logistic93 uses recorded fixed C=1, balanced, max_iter=5000 with fold-local variance filtering/scaling. lbfgs/L2, tol=1e-4 and seed 1729 are implementation choices, not claims about the prior record. No model selection. It is a separate held-out style measure. Historical RF probabilities are a separate held-out evaluator; neither is fully independent evidence because training corpus and custom93 features overlap. E3 outputs were already selected by RMS67, so gains/agreement there are descriptive selection-dependent diagnostics.", "",
        "Event profiles adapt the audited Groove2Groove principle to all piano notes: separate 24x12 onset-duration and 24x41 forward lag/signed interval profiles. They use four quarter-beats in every meter, clip duration overflow, exclude pitch-interval overflow, and symmetrise simultaneous pairs. Cosine is undefined for empty profiles. No BIAB chord/instrument model or exact upstream reproduction is claimed; count normalisation/prototypes weight works equally. This is structurally distinct but shares the MIDI corpus/parser and is not perceptual ground truth.", "",
        "Frozen parser FIFO note pairing and output Skyline reselection (RMS67 only) can differ from the original protected melody. The V2-03 raw unmatched/overlap and exact-pitch measurements remain separate, unchanged artifacts. Content-onset retention correlations are diagnostics only, with undefined values counted; content is never folded into style.", "",
        "Five repeated held-out observations are averaged within original works before uncertainty; they are not independent samples. Bootstrap does not refit models or represent training variation, and there is no multiplicity correction. Corpus ranking is composer discrimination on this corpus, not general musical/perceptual validation. Historical provenance gaps remain unrepaired.", "",
        "See protocol.json, provenance.json, fits/, fit_manifest.json, feature_cache.json, real_scores.json, per_output.json, summary.json, audit.json and frozen_sha256.json. No extraction dependency on musif, generation, optimization, E1c selection or push.", ""]
    return "\n".join(lines)


def run_style_audit(checkout: Path, output: Path, *, configuration: dict | None = None,
                    provenance: dict | None = None, expected_counts=(150, 300), expected_folds=25) -> dict:
    started = time.perf_counter()
    checkout, output = checkout.resolve(), output.resolve()
    roots = resolve_asset_roots(checkout, configuration)
    data, results = (Path(roots[n]["path"]) for n in ("data", "results"))
    previous = [results / f"research_v2_{n:02d}_2026-10-04" for n in (1, 2, 3)]

    def protected_files():
        paths = set(frozen_files(checkout, roots))
        for directory in previous:
            paths.update(p.resolve() for p in directory.rglob("*") if p.is_file()
                         and "musif_env" not in p.parts and "__pycache__" not in p.parts)
        paths.update(p.resolve() for p in (data / "derived/e1_asap").glob("*.json"))
        paths.update(p.resolve() for p in (checkout / "docs/research").glob("V2_0[23]_COMPLETION.md"))
        for name in ("content_metrics.py", "content_audit.py", "feature_feasibility.py", "feature_backends", "asset_paths.py", "provenance.py", "research_audit.py"):
            path = checkout / "src/musicians_style" / name
            if path.is_dir():
                paths.update(p.resolve() for p in path.rglob("*.py"))
            elif path.is_file():
                paths.add(path.resolve())
        for name in ("content_audit.py", "feature_feasibility.py", "feature_backend_worker.py", "research_audit.py"):
            path = checkout / "tools" / name
            if path.is_file():
                paths.add(path.resolve())
        for directory in (checkout / "docs/results", checkout / "requirements", checkout / "tests", checkout / "src"):
            paths.update(p.resolve() for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                         and p.name not in {"style_audit.py", "style_metrics.py", "test_style_audit.py", "test_style_metrics.py"})
        for name in ("requirements.txt", "pyproject.toml", "environment.yml", "pytest.ini"):
            path = checkout / name
            if path.is_file():
                paths.add(path.resolve())
        return paths

    protected = protected_files()
    forbidden = [checkout / "src", checkout / "tools", checkout / "configs", checkout / "docs", data,
                 Path(roots["literature"]["path"]), *previous]
    forbidden += [p for p in results.iterdir() if p.is_dir() and p.name.startswith(("e1_", "e2_", "e3_", "e4_"))] if results.exists() else []
    if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in forbidden):
        raise ValueError("audit destination overlaps frozen/input locations")
    output.mkdir(parents=True, exist_ok=False)
    hashes = {str(p): sha256_file(p) for p in sorted(protected)}
    write_json(output / "protocol.json", CONTRACT)  # before extracting, fitting, or held-out scoring
    write_json(output / "frozen_sha256.json", hashes)
    provenance = provenance or collect_provenance(checkout)
    provenance.update(roots=roots, protocol_fingerprint=fingerprint(CONTRACT),
                      new_source_sha256={str(p.relative_to(checkout)): sha256_file(p) for p in
                          (checkout / "src/musicians_style/style_metrics.py", checkout / "src/musicians_style/style_audit.py", checkout / "tools/style_audit.py", checkout / "docs/research/V2_04_PROTOCOL.md") if p.is_file()})
    write_json(output / "provenance.json", provenance)
    real, records, failures, fits, cache_records, alignment = [], [], [], [], [], []
    models, reprs, custom, events, vectors = {}, {}, {}, {}, {}
    deterministic_scores = True

    def read(path):
        hashes[str(path.resolve())] = sha256_file(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def checked_path(base, relative, expected):
        path = (base / relative.replace("\\", "/")).resolve()
        if not path.is_relative_to(base.resolve()) or not expected or sha256_file(path) != expected:
            raise ValueError("missing, escaped or hash-mismatched input: " + str(path))
        hashes[str(path)] = expected
        return path

    def extract(key, path):
        representation = MidiParser().parse(path)
        e, diagnostics = event_profiles(representation)
        v, names, groups = style_vector(representation)
        c = extract_composition_features(representation)
        if len(v) != 67 or c.shape != (93,) or not np.isfinite(v).all() or not np.isfinite(c).all():
            raise ValueError("invalid representation")
        reprs[key], events[key], vectors[key], custom[key] = representation, e, v, c
        cache_records.append({"key": key, "path": str(path), "sha256": hashes[str(path)], "vector67": v.tolist(),
                              "custom93": c.tolist(), "events": {n: v.tolist() for n, v in e.items()}, "diagnostics": diagnostics,
                              "names67": names, "groups67": groups})

    try:
        if not provenance["import"]["passed"]:
            raise ValueError("checkout import guard failed")
        derived = data / "derived/e1_asap"
        manifest, splits, cached = (read(derived / (n + ".json")) for n in ("manifest", "splits", "composition_features"))
        samples = [r for r in manifest["samples"] if r["validation_status"] == "accepted"]
        if len(samples) != expected_counts[0]:
            raise ValueError("canonical sample count")
        by_id = {r["sample_id"]: r for r in samples}
        folds = validate_splits(samples, splits)
        if len(folds) != expected_folds:
            raise ValueError("frozen fold count")
        frozen_results = {name: read(results / folder / "results.json")["results"] for name, folder in (("E2", "e2_asap"), ("E3", "e3_asap"))}
        pairs = align_tasks(frozen_results["E2"], frozen_results["E3"], samples, folds)
        if len(pairs) != expected_counts[1]:
            raise ValueError("frozen pair count")
        for folder in ("e2_asap", "e3_asap"):
            for name, canonical in (("manifest", manifest), ("splits", splits), ("composition_features", cached)):
                if read(results / folder / "inputs" / (name + ".json")) != canonical:
                    raise ValueError("frozen snapshot mismatch")
        if cached["feature_contract"] != list(FEATURE_SPECS) or cached["features_schema_version"] != COMPOSITION_FEATURES_SCHEMA_VERSION:
            raise ValueError("frozen custom93 schema mismatch")
        cache_map = {r["sample_id"]: r for r in cached["samples"]}
        if len(cache_map) != len(cached["samples"]) or set(cache_map) != set(by_id):
            raise ValueError("frozen custom93 cache coverage")
        dataset = data / Path(manifest["dataset_root"]).name
        for index, r in enumerate(samples):
            key = r["sample_id"]
            extract(key, checked_path(dataset, r["score_path"], r["sha256"]))
            if cache_map[key]["sha256"] != r["sha256"] or not np.array_equal(custom[key], cache_map[key]["values"]):
                raise ValueError("custom93 source/cache mismatch: " + key)
            if (index + 1) % 30 == 0:
                print(f"Source extraction {index + 1}/{len(samples)}", flush=True)
        (output / "fits").mkdir()
        # Complete and persist every fit before scoring any real held-out or transformed output.
        for repeat, fold in folds:
            key = f"r{repeat:02d}_f{fold['fold']:02d}"
            print("Fitting " + key, flush=True)
            model, metadata = fit_measures([by_id[s] for s in fold["train"]["sample_ids"]],
                                          [by_id[s] for s in fold["test"]["sample_ids"]], reprs, custom, events, vectors67=vectors)
            metadata.update(repeat=repeat, fold=fold["fold"], fit_key=key, protocol_fingerprint=fingerprint(CONTRACT))
            model_path = output / "fits" / (key + ".joblib")
            joblib.dump(model, model_path)
            metadata["serialized_model_sha256"] = sha256_file(model_path)
            metadata["fit_seconds_end_since_start"] = time.perf_counter() - started
            write_json(output / "fits" / (key + ".json"), metadata)
            fits.append({"fit_key": key, "model_sha256": metadata["serialized_model_sha256"],
                         "metadata_sha256": sha256_file(output / "fits" / (key + ".json")),
                         "train_samples": len(metadata["train"]), "forbidden_samples": len(metadata["forbidden"]),
                         "warnings": metadata["fit_warnings"]})
            models[(repeat, fold["fold"])] = model
        write_json(output / "fit_manifest.json", fits)
        scoring_started = time.perf_counter() - started
        for repeat, fold in folds:
            fitted = models[(repeat, fold["fold"])]
            reloaded = joblib.load(output / "fits" / f"r{repeat:02d}_f{fold['fold']:02d}.joblib")
            for sid in fold["test"]["sample_ids"]:
                sample = by_id[sid]
                scores = score_measures(fitted, vectors[sid], custom[sid], events[sid])
                deterministic_scores &= scores == score_measures(reloaded, vectors[sid], custom[sid], events[sid])
                ranking = {}
                for m in MEASURES:
                    credit, top = rank_credit(scores[m], sample["composer"])
                    ranking[m] = {"credit": credit, "top_composers": top}
                real.append({"sample_id": sid, "group_id": sample["group_id"], "composer": sample["composer"],
                             "repeat": repeat, "fold": fold["fold"], "fit_key": f"r{repeat:02d}_f{fold['fold']:02d}",
                             "scores": scores, "ranking": ranking})
        real0 = {r["sample_id"]: r for r in real if r["repeat"] == 0}
        content_path = results / "research_v2_03_2026-10-04/content_verified/per_output.json"
        content_contract = read(content_path.parent / "metric_contract.json")
        from .content_metrics import CONTRACT as CONTENT_CONTRACT
        if content_contract != CONTENT_CONTRACT:
            raise ValueError("V2-03 content metric contract mismatch")
        content_audit = read(content_path.parent / "audit.json")
        if not content_audit["passed"] or content_audit["records"] != expected_counts[0] + 2 * expected_counts[1]:
            raise ValueError("V2-03 completed content audit contract failed")
        content_rows = read(content_path)
        content = {(r["experiment"], r["task_id"]): r for r in content_rows}
        if len(content) != len(content_rows):
            raise ValueError("duplicate V2-03 content task")
        for index, (e2, e3) in enumerate(pairs):
            sid, target = e2["source_id"], e2["target_composer"]
            source = by_id[sid]
            fitted = models[(0, e2["fold"])]
            before = real0[sid]["scores"]
            alignment.append({"task_id": e2["task_id"], "source_id": sid, "target": target, "fold": e2["fold"],
                              "E2_sha256": e2["output_sha256"], "E3_sha256": e3["output_sha256"]})
            identity = {"experiment": "identity", "task_id": e2["task_id"], "sample_id": sid,
                        "group_id": source["group_id"], "composer": source["composer"], "target_composer": target,
                        "fit_key": real0[sid]["fit_key"], "scores": before, "byte_identity": True,
                        "movement": {m: movement(before[m], before[m], source["composer"], target) for m in MEASURES}}
            records.append(identity)
            for name, frozen in (("E2", e2), ("E3", e3)):
                key = name + ":" + frozen["task_id"]
                path = checked_path(results / ("e2_asap" if name == "E2" else "e3_asap"), frozen["output_path"], frozen["output_sha256"])
                extract(key, path)
                after = score_measures(fitted, vectors[key], custom[key], events[key])
                if name == "E3" and fitted["profiles"][target].fingerprint != frozen["profile_fingerprint"]:
                    raise ValueError("historical E3 fit fingerprint mismatch")
                c = content[(name, frozen["task_id"])]
                if c.get("output_sha256") != frozen["output_sha256"] or c["source_sha256"] != source["sha256"]:
                    raise ValueError("V2-03 content alignment mismatch")
                records.append({"experiment": name, "task_id": frozen["task_id"], "sample_id": sid,
                    "group_id": source["group_id"], "composer": source["composer"], "target_composer": target,
                    "fit_key": real0[sid]["fit_key"], "source_sha256": source["sha256"], "output_sha256": frozen["output_sha256"],
                    "scores": after, "byte_identity": frozen["output_sha256"] == source["sha256"],
                    "historical_identity_genome": name == "E3" and all(v == 0 for v in frozen["best_genome"].values()),
                    "historical_objective": frozen.get("style_gain") if name == "E3" else None,
                    "historical_rf_delta": frozen["delta_p_target"],
                    "content_on_fraction": c.get("measurement", {}).get("v2_exact_pitch", {}).get("on_event_retention_fraction"),
                    "content_event_status": c.get("measurement", {}).get("v2_exact_pitch", {}).get("event_identity_status", "undefined"),
                    "movement": {m: movement(before[m], after[m], source["composer"], target) for m in MEASURES}})
                # Retain feature arrays, release long note representations once scored.
                del reprs[key]
            if (index + 1) % 30 == 0:
                print(f"Aligned pair scoring {index + 1}/{len(pairs)}", flush=True)
        provenance["held_out_scoring_started_seconds"] = scoring_started
        write_json(output / "provenance.json", provenance)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    print("Summarising clustered evidence", flush=True)
    summary = summarize(real, records) if len(records) == 3 * expected_counts[1] else {}
    changed = [p for p, h in hashes.items() if not Path(p).is_file() or sha256_file(Path(p)) != h]
    added = sorted(str(p) for p in protected_files() if str(p) not in hashes)
    rms_differences = [r["movement"]["rms67"]["delta"] - r["historical_objective"] for r in records if r["experiment"] == "E3"]
    checks = {"all_real_scores": len(real) == expected_counts[0] * (expected_folds // 5),
              "all_300_pairs_and_identities": len(records) == 3 * expected_counts[1] and len(alignment) == expected_counts[1],
              "all_fits_before_scoring": len(fits) == expected_folds,
              "serialized_fit_scores_exact": deterministic_scores and bool(real),
              "frozen_and_completed_v2_preserved": not changed and not added,
              "historical_rms_reproduction": bool(rms_differences) and max(map(abs, rms_differences)) < 1e-10}
    if summary:
        summary["event_diagnostics"] = {}
        for experiment in ("source", "E2", "E3"):
            rows = [r for r in cache_records if (r["key"].startswith(experiment + ":") if experiment != "source" else ":" not in r["key"])]
            summary["event_diagnostics"][experiment] = {"pieces": len(rows), **{field: sum(r["diagnostics"][field] for r in rows) for field in
                ("notes", "duration_ge_2", "duration_nonpositive", "time_pitch_pairs", "time_pitch_excluded_pitch_pairs", "non_4_4", "meter_changes", "meter_missing")},
                "empty_profiles": dict(Counter(n for r in rows for n in r["diagnostics"]["empty_profiles"]))}
        summary["fit_diagnostics"] = {"fits": len(fits), "warnings": [w for r in fits for w in r["warnings"]],
                                      "serialized_score_determinism": deterministic_scores}
    audit = {"schema": "v2-04.audit.1", "passed": not failures and all(checks.values()), "checks": checks,
             "failures": failures, "real_rows": len(real), "output_rows": len(records), "aligned_pairs": len(alignment),
             "protected_files": len(hashes), "changed_or_missing": changed, "added": added,
             "max_abs_historical_rms_difference": max(map(abs, rms_differences)) if rms_differences else None,
             "elapsed_seconds": time.perf_counter() - started}
    for filename, value in (("feature_cache", cache_records), ("real_scores", real), ("per_output", records),
                            ("alignment", alignment), ("summary", summary), ("audit", audit), ("frozen_sha256", hashes)):
        write_json(output / (filename + ".json"), value)
    (output / "report.md").write_text(report(summary, audit) if summary else "# Failed V2-04 audit\n\n" + "\n".join(failures), encoding="utf-8")
    return audit
