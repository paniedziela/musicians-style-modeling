"""Bounded V2-05 pilot; isolated bindings to the unchanged frozen E3 engine."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import FunctionType

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from .e3 import algorithm
from .e3.algorithm import E3GeneticAlgorithm, SearchConfig
from .e3.objective import validate_constraints
from .e3.profile import style_vector
from .e3.structure import analyse_structure
from .e3.types import CandidateEvaluation, E3Genome
from .e1.composition_features import extract_composition_features
from .asset_paths import resolve_asset_roots
from .content_metrics import CONTRACT as CONTENT_CONTRACT, measure_content, observe_midi
from .midi.printer import MidiPrettyPrinter
from .provenance import collect_provenance, fingerprint, sha256_file, write_json
from .style_audit import movement, validate_splits
from .style_metrics import CONTRACT as STYLE_CONTRACT, MEASURES, assert_disjoint, event_profiles, fit_measures, score_measures

OBJECTIVES = ("rms67", "gaussian67", "logistic93")
SEARCH = SearchConfig(population_size=32, generations=60, elitism_k=2, tournament_size=3,
                      stagnation_generations=61, mutation_sigma=(1., .12, .12, .12, .12))
SEED = 1729
PROTOCOL = {
    "schema": "v2-05.pilot.1", "objectives": OBJECTIVES, "seed": SEED,
    "search": asdict(SEARCH), "repeat": 0, "outer_fold": 0, "inner_fold": 0,
    "selection": "first lexical (group_id, sample_id) per composer in frozen r00/f00/inner00 validation; no score/length/convenience inspection",
    "cohort": "3 fixed validation sources, six directions, three objectives, 18 runs",
    "budget": {"initial_population": 32, "new_offspring": 60 * 30,
               "proposal_budget": 32 + 60 * 30, "evaluation_requests_including_elites_and_identity": 1 + 61 * 32,
               "history_rows": 61, "actual_unique_evaluations": "unique transformed representations evaluated, including infeasible candidates"},
    "adapter": "execute original E3 run code with a private globals copy binding _clamp to frozen clamp followed by transpose=0; no shared monkeypatch or frozen source edit; all five RNG draws retained",
    "content": "original protected absolute pitches/onsets/offs/identifiable durations; zero transpose before every transformation/evaluation; same-tick order diagnostic",
    "constraints": "frozen E3 constraints plus V2-03 serialized event/content/structural checks; ambiguous overlapping durations reported, never labelled proven exact",
    "objective_scoring": "higher-is-better target affinity on saved-MIDI representation minus original source affinity; identity uses original bytes",
    "style_contract": STYLE_CONTRACT, "content_contract": CONTENT_CONTRACT,
    "fitting": "one fixed inner-training bundle for all cases; validation and outer test forbidden by sample/group/hash; no pilot-output fitting or calibration",
    "rf": {"pipeline": "VarianceThreshold(0), RF", "n_estimators": 300, "max_features": .5,
           "min_samples_leaf": 2, "random_state": SEED, "class_weight": "balanced", "n_jobs": 1,
           "provenance": "historical E3 evaluator configuration refit on the same inner training only; shared corpus/custom93; not independent"},
    "evaluation": "all five V2-04 measures, RF, optimized gain, original-mask V2-03 content, structure, runtime, unique counts, convergence; no combined scalar",
    "dependence": "RMS/Gaussian share 67 features; logistic is non-independent when optimized and shares custom93/corpus with RF; time_pitch primary distinct diagnostic, onset_duration secondary",
    "null": "abs(delta)<=1e-12; negative delta<-1e-12",
    "determinism": "synthetic same-seed repeated adapter tests; exact persisted-fit scores and repeat serialization for all real outputs; no additional real optimization runs",
    "inference": "exploratory feasibility only, one source/work per composer; no confirmatory tests or candidate tuning",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def select_cohort(samples, splits):
    folds = validate_splits(samples, splits)  # identities only; never read outer-test MIDI/features
    fold = next(f for r, f in folds if (r, f["fold"]) == (0, 0))
    inner = next(f for f in fold["inner_folds"] if f["fold"] == 0)
    by_id = {r["sample_id"]: r for r in samples}
    train = [by_id[s] for s in sorted(inner["train"]["sample_ids"])]
    validation = [by_id[s] for s in inner["validation"]["sample_ids"]]
    forbidden = [by_id[s] for s in sorted(set(by_id) - {r["sample_id"] for r in train})]
    assert_disjoint(train, forbidden)
    composers = sorted({r["composer"] for r in samples})
    cohort = [min((r for r in validation if r["composer"] == c),
                  key=lambda r: (r["group_id"], r["sample_id"])) for c in composers]
    if len(cohort) != 3 or len({r["group_id"] for r in cohort}) != 3:
        raise ValueError("pilot requires three distinct source composers/works")
    return cohort, train, forbidden


def content_failures(metrics):
    p = metrics["v2_exact_pitch"]
    failures = ["protected_events"] if p["event_identity_status"] != "passed" else []
    if p["duration_status"] in ("failed", "undefined"):
        failures.append("protected_duration")
    failures.extend(k for k, v in metrics["structural_technical"].items() if v is not True)
    if p["protected_velocity_status"] != "passed":
        failures.append("protected_velocity")
    return failures


def objective_affinity(fitted, name, target, piece):
    """Reuse V2-04 scoring; event profiles are intentionally undefined during search."""
    if name not in OBJECTIVES:
        raise ValueError("not a predeclared pilot objective")
    scores = score_measures(fitted, style_vector(piece)[0], extract_composition_features(piece),
                            {"onset_duration": np.zeros(288), "time_pitch": np.zeros(984)})
    return scores[name][target]


class PilotObjective:
    def __init__(self, source_observation, fitted, name, target):
        if name not in OBJECTIVES:
            raise ValueError("unsupported pilot objective")
        self.observation = source_observation
        self.source = source_observation.piece
        self.structure = analyse_structure(self.source)
        self.profile = fitted["profiles"][target]
        self.fitted, self.name, self.target = fitted, name, target
        self.baseline = objective_affinity(fitted, name, target, self.source)
        self.evaluations = 0
        self.rejections = Counter()

    def evaluate(self, genome, output):
        if genome.transpose_semitones != 0:
            raise AssertionError("transposition reached objective")
        self.evaluations += 1
        constraints = validate_constraints(self.source, output, self.profile, genome,
                                           source_structure=self.structure, roundtrip=True)
        gain = 0.0
        if constraints.feasible:
            observation = self.observation if output == self.source else observe_midi(MidiPrettyPrinter().to_bytes(output))
            extra = content_failures(measure_content(self.observation, observation))
            if extra:
                constraints = replace(constraints, feasible=False,
                                      violations=constraints.violations + tuple(extra),
                                      violation_score=constraints.violation_score + len(extra))
            else:
                gain = objective_affinity(self.fitted, self.name, self.target, observation.piece) - self.baseline
        self.rejections.update(constraints.violations)
        return CandidateEvaluation(genome, output, float(gain), {self.name: float(gain)}, constraints)


def run_search(source, profile, objective, config=SEARCH, seed=SEED, progress=None):
    """Rebind only canonicalization in a private function; frozen globals remain untouched."""
    telemetry = Counter()
    frozen_clamp = algorithm._clamp
    frozen_transform = algorithm.apply_transformation

    def canonical(genome):
        telemetry["canonicalization_calls"] += 1
        telemetry["nonzero_gene_discarded"] += int(round(genome.transpose_semitones) != 0)
        return replace(frozen_clamp(genome), transpose_semitones=0)

    def transform(source, genome, profile, **kwargs):
        if genome.transpose_semitones != 0:
            raise AssertionError("transposition reached transformation")
        telemetry["transformations"] += 1
        return frozen_transform(source, genome, profile, **kwargs)

    original = E3GeneticAlgorithm.run
    bindings = dict(original.__globals__, _clamp=canonical, apply_transformation=transform)
    bound = FunctionType(original.__code__, bindings, original.__name__, original.__defaults__, original.__closure__)
    bound.__kwdefaults__ = original.__kwdefaults__
    result = bound(E3GeneticAlgorithm(config), source, profile, seed=seed, objective=objective, progress=progress)
    if algorithm._clamp is not frozen_clamp or algorithm.apply_transformation is not frozen_transform:
        raise AssertionError("frozen engine globals changed")
    if result.genome.transpose_semitones or any(r["best_genome"]["transpose_semitones"] for r in result.history):
        raise AssertionError("nonzero saved/history genome")
    return result, dict(telemetry)


def summarize(rows):
    result = {}
    for name in OBJECTIVES:
        subset = [r for r in rows if r["objective"] == name and r["status"] == "completed"]
        measures = {}
        for measure in (*MEASURES, "historical_rf"):
            deltas = [r["movement"][measure]["delta"] for r in subset]
            finite = [v for v in deltas if v is not None]
            measures[measure] = dict(defined=len(finite), undefined=len(deltas)-len(finite),
                mean=float(np.mean(finite)) if finite else None,
                positive=sum(v > 1e-12 for v in finite), negative=sum(v < -1e-12 for v in finite),
                null=sum(abs(v) <= 1e-12 for v in finite),
                directions={r["direction"]: r["movement"][measure]["delta"] for r in subset})
        result[name] = dict(completed=len(subset), measures=measures,
            content_verified=sum(r["content_policy_verified"] for r in subset),
            ambiguous_duration=sum(r["content"]["v2_exact_pitch"]["duration_status"] == "ambiguous" for r in subset),
            identity_outputs=sum(r["source_sha256"] == r["output_sha256"] for r in subset),
            runtime_seconds=sum(r["elapsed_seconds"] for r in subset),
            unique_evaluations=sum(r["unique_evaluations"] for r in subset),
            disagreement_with_optimized={m: sum((r["optimized_gain"] > 1e-12) and
                 (r["movement"][m]["delta"] is not None) and (r["movement"][m]["delta"] <= 1e-12) for r in subset)
                 for m in (*MEASURES, "historical_rf")})
    return result


def report(rows, summary, audit):
    lines = ["# V2-05 fixed-budget objective pilot", "", f"Execution/preservation passed: {audit['passed']}. Runtime: {audit['elapsed_seconds']:.2f} s.", "",
        "Three lexical validation sources from r00/f00/inner00; all six directions per objective. All fits use only the 76 inner-training pieces. No outer-test MIDI/features are opened for fitting, selection or scoring.", "",
        "Equal budget: 32 population, 60 generations, 61 stagnation limit, seed 1729; 1,832 proposals and 1,953 evaluation requests per run. Elitism 2, tournament 3, all five historical mutation draws retained, transposition always canonicalized to zero before transformation/evaluation.", "",
        "## Separate style movement", "", "| Objective | Measure | Mean delta | Positive / negative / null / undefined |", "|---|---|---:|---|"]
    for o, s in summary.items():
        for m, v in s["measures"].items():
            lines.append(f"| {o} | {m} | {v['mean']} | {v['positive']} / {v['negative']} / {v['null']} / {v['undefined']} |")
    lines += ["", "## Six directions and content", "", "| Objective | Direction | Optimized gain | time_pitch | onset_duration | logistic93 | RF | Content | Unique evaluations | Seconds |", "|---|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in rows:
        if r["status"] != "completed":
            lines.append(f"| {r['objective']} | {r['direction']} | FAILED: {r['error']} | | | | | | | |")
            continue
        d = r["movement"]
        lines.append(f"| {r['objective']} | {r['direction']} | {r['optimized_gain']:.6g} | {d['time_pitch']['delta']} | {d['onset_duration']['delta']} | {d['logistic93']['delta']} | {d['historical_rf']['delta']} | {r['content_policy_verified']}; duration {r['content']['v2_exact_pitch']['duration_status']} | {r['unique_evaluations']} | {r['elapsed_seconds']:.2f} |")
    lines += ["", "## Interpretation and limits", "",
        "Evaluate exact content first, then structurally distinct time_pitch movement, six-direction consistency, null/negative rates and disagreements. Optimized gain alone is insufficient. summary.json contains all directions and explicit optimized/non-optimized disagreement counts. No style/content combined score, significance test or confirmatory confidence interval is calculated for three sources.", "",
        "RMS/Gaussian share the 67-feature representation and differ in link and variance handling. Logistic93 is explicitly non-independent for logistic-optimized outputs; it and the historical-configuration RF share custom93 and training corpus. RF is freshly fit on inner training, not reused from an outer-training model that saw validation. Event measures share MIDI/parser/corpus but change representation; time_pitch is primary, onset_duration secondary. No perceptual or listening-study conclusion.", "",
        "V2-03 content diagnostics retain their historical strict-order fields; the pilot hard policy excludes same-tick order. Ambiguous overlapping voice durations remain ambiguous, not certified exact. Original-mask observable on/off retention, zero transposition, frozen tuple/velocity constraints, structural invariants and serialized measurements are separate. Identity outputs retain original bytes. Skyline reselection is diagnostic.", "",
        "The unchanged frozen search code runs with a private canonicalization binding. No global monkeypatch, new operator or frozen E3 hook/edit. All five RNG draws are consumed; inactive transpose cannot affect transformation, cache keys or evaluations. Actual unique evaluations vary with objective and cache collisions despite identical proposal budgets.", "",
        "Determinism is tested with repeated synthetic search for all three objectives, exact joblib score round trips and repeated MIDI serialization of each selected real output. The 18 real searches are not repeated; full real same-seed rerun reproducibility remains unmeasured. No pilot outputs are used to fit/tune/select/calibrate any metric. This is exploratory feasibility evidence; stop for review after V2-05.", "",
        "Artifacts: pilot_manifest.json, protocol.json, provenance.json, fit_manifest.json, fits/, per_output.json, summary.json, audit.json, frozen_sha256.json, tasks/*/output.mid and result.json/history.json. No push or larger experiment."]
    return "\n".join(lines) + "\n"


def run_pilot(checkout: Path, output: Path, *, configuration=None, provenance=None):
    started = time.perf_counter()
    checkout, output = checkout.resolve(), output.resolve()
    roots = resolve_asset_roots(checkout, configuration)
    data, results = (Path(roots[n]["path"]) for n in ("data", "results"))
    if not output.is_relative_to(results) or output == results or output.exists():
        raise ValueError("fresh child of results root required")
    if any(output.is_relative_to(p) for p in results.iterdir() if p.is_dir()):
        raise ValueError("destination overlaps existing experiment")
    output.mkdir(parents=True)
    frozen = set()
    for directory in (checkout / "src", checkout / "configs", checkout / "requirements", checkout / "docs/results", checkout / "tools", checkout / "tests"):
        frozen.update(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name not in {"objective_pilot.py", "test_objective_pilot.py"})
    for directory in results.iterdir():
        if directory.is_dir() and directory != output and directory.name.startswith(("e1_", "e2_", "e3_", "e4_", "research_v2_01_", "research_v2_02_", "research_v2_03_", "research_v2_04_")):
            frozen.update(p for p in directory.rglob("*") if p.is_file() and "musif_env" not in p.parts and "__pycache__" not in p.parts)
    frozen.update((checkout / "docs/research").glob("V2_0[234]_*.md"))
    input_dir = data / "derived/e1_asap"
    manifest_path, splits_path = input_dir / "manifest.json", input_dir / "splits.json"
    frozen.update((manifest_path, splits_path))
    hashes = {str(p): sha256_file(p) for p in sorted(frozen)}
    write_json(output / "frozen_sha256.json", hashes)
    write_json(output / "protocol.json", PROTOCOL)
    samples = [r for r in json.loads(manifest_path.read_text(encoding="utf-8"))["samples"] if r.get("validation_status") == "accepted"]
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    cohort, train, forbidden = select_cohort(samples, splits)
    source_root = data / "asap-dataset-1.2"
    accessed = sorted(train + cohort, key=lambda r: r["sample_id"])
    for r in accessed:
        path = (source_root / r["score_path"]).resolve()
        if not path.is_relative_to(source_root.resolve()) or sha256_file(path) != r["sha256"]:
            raise ValueError("source path/hash mismatch: " + r["sample_id"])
        hashes[str(path)] = r["sha256"]
    tasks = [dict(task_id=f"{s['sample_id']}__to__{target}__{name}", source_id=s["sample_id"], source_composer=s["composer"],
                  target=target, objective=name, direction=s["composer"] + "->" + target)
             for s in cohort for target in sorted({r["composer"] for r in cohort} - {s["composer"]}) for name in OBJECTIVES]
    write_json(output / "pilot_manifest.json", dict(persisted_at=now(), protocol_fingerprint=fingerprint(PROTOCOL),
        cohort=cohort, train=train, forbidden_identities=[{k:r[k] for k in ("sample_id", "composer", "group_id", "sha256")} for r in forbidden],
        tasks=tasks, input_hashes={str(p):sha256_file(p) for p in (manifest_path, splits_path)}, accessed_source_ids=[r["sample_id"] for r in accessed]))
    write_json(output / "frozen_sha256.json", hashes)
    provenance = provenance or collect_provenance(checkout)
    provenance.update(roots=roots, protocol_fingerprint=fingerprint(PROTOCOL),
        source_hashes={str(p.relative_to(checkout)):sha256_file(p) for p in (checkout / "src/musicians_style/objective_pilot.py", checkout / "tools/objective_pilot.py", checkout / "docs/research/V2_05_PROTOCOL.md")},
        outer_test_access="identity metadata only for disjointness; no outer-test original MIDI/features opened; old scientific artifacts hashed only for preservation")
    write_json(output / "provenance.json", provenance)
    observations = {r["sample_id"]: observe_midi(source_root / r["score_path"]) for r in accessed}
    reprs = {s:o.piece for s,o in observations.items()}
    custom = {s:extract_composition_features(p) for s,p in reprs.items()}
    events = {s:event_profiles(p)[0] for s,p in reprs.items()}
    fitted, metadata = fit_measures(train, forbidden, reprs, custom, events)
    rf = Pipeline([("variance", VarianceThreshold()), ("model", RandomForestClassifier(**{k:v for k,v in PROTOCOL['rf'].items() if k not in ('pipeline','provenance')}))])
    rf.fit(np.asarray([custom[r["sample_id"]] for r in train]), [r["composer"] for r in train])
    metadata["historical_rf"] = dict(parameters=PROTOCOL["rf"], train=metadata["train"], forbidden=metadata["forbidden"], classes=rf.classes_.tolist())
    fits = output / "fits"
    fits.mkdir()
    joblib.dump(dict(style=fitted, rf=rf), fits / "inner00.joblib")
    write_json(fits / "inner00.json", metadata)
    restored = joblib.load(fits / "inner00.joblib")
    for s in reprs:
        a = score_measures(fitted, style_vector(reprs[s])[0], custom[s], events[s])
        b = score_measures(restored["style"], style_vector(reprs[s])[0], custom[s], events[s])
        if a != b or not np.array_equal(rf.predict_proba(custom[s].reshape(1,-1)), restored["rf"].predict_proba(custom[s].reshape(1,-1))):
            raise ValueError("serialized fit score mismatch")
    write_json(output / "fit_manifest.json", dict(persisted_at=now(), fit_sha256={str(p.relative_to(output)):sha256_file(p) for p in fits.iterdir()},
        fit_warnings=metadata["fit_warnings"], exact_serialized_scores=True, training_count=len(train), forbidden_count=len(forbidden)))
    write_json(output / "optimization_started.json", dict(timestamp=now(), manifest_sha256=sha256_file(output / "pilot_manifest.json"),
        fit_manifest_sha256=sha256_file(output / "fit_manifest.json"), protocol_sha256=sha256_file(output / "protocol.json")))
    rows = []
    for task in tasks:
        tick = time.perf_counter()
        task_dir = output / "tasks" / task["task_id"]
        task_dir.mkdir(parents=True)
        row = dict(task, status="failed")
        try:
            sid, target, name = task["source_id"], task["target"], task["objective"]
            obs = observations[sid]
            objective = PilotObjective(obs, fitted, name, target)
            result, telemetry = run_search(obs.piece, fitted["profiles"][target], objective)
            byte_output = (source_root / next(r["score_path"] for r in cohort if r["sample_id"] == sid)).read_bytes() if result.output == obs.piece else MidiPrettyPrinter().to_bytes(result.output)
            serial_equal = result.output == obs.piece or byte_output == MidiPrettyPrinter().to_bytes(result.output)
            output_path = task_dir / "output.mid"
            output_path.write_bytes(byte_output)
            after = observe_midi(byte_output)
            content = measure_content(obs, after)
            p = after.piece
            scores_in = score_measures(fitted, style_vector(obs.piece)[0], custom[sid], events[sid])
            event_after, event_diagnostics = event_profiles(p)
            custom_after = extract_composition_features(p)
            scores_out = score_measures(fitted, style_vector(p)[0], custom_after, event_after)
            for scores, vector in ((scores_in, custom[sid]), (scores_out, custom_after)):
                scores["historical_rf"] = dict(zip(rf.classes_, map(float,rf.predict_proba(vector.reshape(1,-1))[0])))
            delta = {m:movement(scores_in[m], scores_out[m], task["source_composer"], target) for m in scores_in}
            if abs(delta[name]["delta"] - result.evaluation.style_gain) > 1e-12:
                raise ValueError("optimized versus saved-output movement mismatch")
            if result.unique_candidates != objective.evaluations or len(result.history) != 61 or result.unique_candidates + result.cache_hits != 1953:
                raise ValueError("search budget/evaluation accounting mismatch")
            source_row = next(r for r in cohort if r["sample_id"] == sid)
            row.update(status="completed", source_sha256=source_row["sha256"], output_sha256=sha256_file(output_path),
                group_id=source_row["group_id"], genome=asdict(result.genome), optimized_gain=result.evaluation.style_gain,
                input_scores=scores_in, output_scores=scores_out, movement=delta, content=content,
                content_policy_verified=not content_failures(content), constraints=asdict(result.evaluation.constraints),
                event_diagnostics=event_diagnostics, stop_reason=result.stop_reason, unique_evaluations=result.unique_candidates,
                cache_hits=result.cache_hits, proposal_budget=1832, evaluation_requests=1953, telemetry=telemetry,
                rejected_constraint_counts=dict(objective.rejections), serialization_deterministic=serial_equal,
                evaluator_dependence=dict(logistic93_non_independent=name == "logistic93", rf="shared inner corpus/custom93", time_pitch="primary structurally distinct", onset_duration="secondary event diagnostic"))
            write_json(task_dir / "history.json", list(result.history))
        except Exception as exc:
            row.update(error=type(exc).__name__ + ": " + str(exc))
        row["elapsed_seconds"] = time.perf_counter() - tick
        write_json(task_dir / "result.json", row)
        rows.append(row)
        write_json(output / "per_output.json", rows)
        print(f"{len(rows)}/18 {task['direction']} {task['objective']}: {row['status']} ({row['elapsed_seconds']:.2f}s)", flush=True)
    changed = [p for p,h in hashes.items() if not Path(p).is_file() or sha256_file(Path(p)) != h]
    added = []
    for directory in results.iterdir():
        if directory.is_dir() and directory != output and directory.name.startswith(("e1_", "e2_", "e3_", "e4_", "research_v2_01_", "research_v2_02_", "research_v2_03_", "research_v2_04_")):
            added.extend(str(p) for p in directory.rglob("*") if p.is_file() and "musif_env" not in p.parts and "__pycache__" not in p.parts and str(p) not in hashes)
    audit = dict(passed=len(rows)==18 and all(r["status"]=="completed" and r["content_policy_verified"] and r["constraints"]["feasible"] and r["serialization_deterministic"] for r in rows) and not changed and not added,
        elapsed_seconds=time.perf_counter()-started, runs=len(rows), completed=sum(r["status"]=="completed" for r in rows),
        protected_files=len(hashes), changed_protected=changed, added_protected=added,
        accessed_original_sources=[r["sample_id"] for r in accessed], outer_test_original_source_access_count=0)
    summary = summarize(rows)
    write_json(output / "summary.json", summary)
    write_json(output / "audit.json", audit)
    (output / "report.md").write_text(report(rows, summary, audit), encoding="utf-8")
    return audit
