"""Concrete, fold-local E1d measures; no search or metric selection."""

from __future__ import annotations

from collections import defaultdict
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .e3.objective import grouped_distance
from .e3.profile import GROUPS, build_target_profile, style_vector

MEASURES = ("rms67", "gaussian67", "logistic93", "onset_duration", "time_pitch")
LOGISTIC = dict(C=1.0, class_weight="balanced", solver="lbfgs", penalty="l2",
                max_iter=5000, tol=1e-4, random_state=1729)
CONTRACT = {
    "schema": "v2-04.style.1", "direction": "higher affinity is closer; movement = output minus source",
    "rms67": "negative equal-family mean of frozen E3 pitch/rhythm/texture RMS z distances",
    "gaussian67": "equal-family mean of mean exp(-z^2/2) marginal criterion satisfactions; not a density or likelihood",
    "rms_variance": "frozen E3 per-composer population std; std < 1e-6 replaced by 1.0",
    "gaussian_variance": "recorded V2 thesis adaptation, training only: max(target-composer population std, 0.05 * pooled-training population std, 1e-6); different from frozen E3",
    "logistic93": {"affinity": "predict_proba(composer); separate held-out style measure, shared corpus/features with RF",
                   "pipeline": "VarianceThreshold(0), StandardScaler, LogisticRegression", "parameters": LOGISTIC,
                   "recorded_settings": "C=1, balanced, training-only variance/scaling, max_iter=5000; no selection",
                   "implementation_choices": "lbfgs/L2, tol=1e-4, seed 1729 (solver not fixed in prior V2 record)"},
    "onset_duration": "all parsed piano notes, 24x12 histogram; onset quarter-beats modulo 4, duration clipped into [0,2]; last bin includes >=2",
    "time_pitch": "all parsed piano note pairs at forward onset lag [0,4) quarter-beats, 24x41 histogram, signed later-minus-earlier pitch [-20,20]; out-of-range excluded; simultaneous unordered pairs emit both signs",
    "event_similarity": "cosine to per-composer equal-work mean of per-piece L1-normalised profiles; empty query/prototype undefined",
    "event_meter": "fixed four-quarter-beat windows in every meter, no rescaling; non-4/4 and changes diagnosed",
    "event_adaptation": "L0074 audited event-profile principle; all piano notes, no BIAB chords/instruments, no claim of exact upstream reproduction; no velocity",
    "ranking": "all 3 composers, max affinity; abs difference <=1e-12 is a tie; fractional correct credit 1/k for k tied maxima; all-score missing rows undefined",
    "null": "abs(movement)<=1e-12; negative movement < -1e-12; fixed numeric tolerance, no tuned threshold",
    "uncertainty": "2000 percentile bootstrap draws, seed 1729, work groups; held-out ranking averages samples/repeats within work then macro composers, resamples works within composer",
    "agreement": "Pearson and Spearman on complete pairs; primary work-mean pairs with work bootstrap CI, output-level coefficients also retained; constant/range<=1e-12/<3 pairs undefined; real agreement on true-composer affinity and separately each candidate composer",
    "promotion": "review evidence on ranking with CI versus 1/3, score spread/missingness, direction consistency, agreement structure, leakage and interpretation; no automatic objective selection",
    "scope": "25 frozen outer folds for real works; only matching repeat-0 folds for 300 aligned E2/E3 pairs; no inner selection, musif, transfer or composite",
}


def assert_disjoint(train: list[dict], forbidden: list[dict]) -> None:
    """Check every identity before any learned operation, including scaling."""
    for rows in (train, forbidden):
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate sample identity")
        if any(not r.get(k) for r in rows for k in ("sample_id", "group_id", "sha256", "composer")):
            raise ValueError("missing fit identity/provenance")
    for field in ("sample_id", "group_id", "sha256"):
        if {r[field] for r in train} & {r[field] for r in forbidden}:
            raise ValueError(field + " forbidden overlap")


def event_profiles(repr_) -> tuple[dict[str, np.ndarray], dict]:
    if repr_.ticks_per_beat <= 0:
        raise ValueError("invalid tick resolution")
    notes = sorted(repr_.notes, key=lambda n: (n.tick, n.pitch, n.channel, n.duration_ticks))
    od = np.zeros((24, 12), dtype=float)
    tp = np.zeros((24, 41), dtype=float)
    ticks = np.asarray([n.tick for n in notes])
    pitches = np.asarray([n.pitch for n in notes])
    excluded = pairs = long = nonpositive = 0
    for i, note in enumerate(notes):
        duration = note.duration_ticks / repr_.ticks_per_beat
        long += int(duration >= 2)
        nonpositive += int(duration <= 0)
        od[min(23, int((note.tick / repr_.ticks_per_beat % 4) * 6)),
           min(11, max(0, int(duration * 6)))] += 1
        end = int(np.searchsorted(ticks, note.tick + 4 * repr_.ticks_per_beat, side="left"))
        lag = (ticks[i + 1:end] - note.tick) / repr_.ticks_per_beat
        delta = pitches[i + 1:end] - note.pitch
        # Equal-onset pairs have no intrinsic temporal orientation.
        delta = np.concatenate([delta, -delta[lag == 0]])
        lag = np.concatenate([lag, lag[lag == 0]])
        pairs += len(delta)
        valid = np.abs(delta) <= 20
        excluded += int((~valid).sum())
        np.add.at(tp, (np.minimum(23, (lag[valid] * 6).astype(int)), delta[valid] + 20), 1)
    vectors = {}
    for name, counts in (("onset_duration", od), ("time_pitch", tp)):
        vector = counts.ravel()
        vectors[name] = vector / vector.sum() if vector.sum() else vector
    meters = [m.payload for m in repr_.meta if m.kind == "time_signature"]
    return vectors, {"notes": len(notes), "duration_ge_2": long, "duration_nonpositive": nonpositive,
                     "time_pitch_pairs": pairs, "time_pitch_excluded_pitch_pairs": excluded,
                     "empty_profiles": [k for k, v in vectors.items() if not v.any()],
                     "meters": meters, "non_4_4": any(m.get("numerator") != 4 or m.get("denominator") != 4 for m in meters),
                     "meter_changes": len(meters) > 1, "meter_missing": not meters,
                     "parser_caution": "frozen FIFO note pairing; raw unmatched/overlap cases separately recorded by V2-03"}


def cosine(vector: np.ndarray, prototype: np.ndarray) -> float | None:
    denominator = np.linalg.norm(vector) * np.linalg.norm(prototype)
    if not denominator or not np.isfinite(denominator):
        return None
    return float(np.clip(np.dot(vector, prototype) / denominator, -1, 1))


def fit_measures(train: list[dict], forbidden: list[dict], representations: dict,
                 custom: dict, events: dict, *, vectors67: dict | None = None) -> tuple[dict, dict]:
    assert_disjoint(train, forbidden)
    train = sorted(train, key=lambda r: r["sample_id"])
    composers = sorted({r["composer"] for r in train})
    profiles = {c: build_target_profile(c, train, representations, forbidden_rows=forbidden) for c in composers}
    training_vectors = {r["sample_id"]: (vectors67[r["sample_id"]] if vectors67 is not None else
                                         style_vector(representations[r["sample_id"]])[0]) for r in train}
    matrix67 = np.asarray([training_vectors[r["sample_id"]] for r in train])
    if matrix67.shape != (len(train), 67) or not np.isfinite(matrix67).all():
        raise ValueError("invalid training 67-component matrix")
    raw_by_composer = {c: np.asarray([training_vectors[r["sample_id"]] for r in train if r["composer"] == c])
                       for c in composers}
    for c, raw in raw_by_composer.items():
        expected_rms_std = np.where(raw.std(axis=0) < 1e-6, 1.0, raw.std(axis=0))
        if not np.array_equal(raw.mean(axis=0), profiles[c].mean) or not np.array_equal(expected_rms_std, profiles[c].std):
            raise ValueError("cached 67-component representation differs from frozen E3")
    pooled_std = matrix67.std(axis=0)
    gaussian_std = {c: np.maximum.reduce([raw_by_composer[c].std(axis=0), 0.05 * pooled_std, np.full(67, 1e-6)])
                    for c in composers}
    prototypes = {}
    for c in composers:
        grouped = defaultdict(list)
        for r in train:
            if r["composer"] == c:
                grouped[r["group_id"]].append(r["sample_id"])
        prototypes[c] = {name: np.mean([np.mean([events[s][name] for s in ids], axis=0)
                                        for ids in grouped.values()], axis=0)
                         for name in ("onset_duration", "time_pitch")}
    x = np.asarray([custom[r["sample_id"]] for r in train])
    if x.shape != (len(train), 93) or not np.isfinite(x).all():
        raise ValueError("invalid frozen custom93 training matrix")
    logistic = Pipeline([("variance", VarianceThreshold()), ("scale", StandardScaler()),
                         ("model", LogisticRegression(**LOGISTIC))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        logistic.fit(x, [r["composer"] for r in train])
    metadata = {"train": [{k: r[k] for k in ("sample_id", "group_id", "sha256", "composer")} for r in train],
                "forbidden": [{k: r[k] for k in ("sample_id", "group_id", "sha256", "composer")} for r in forbidden],
                "leakage_checks": ["sample_id", "group_id", "sha256"], "profiles": {},
                "event_prototypes": {c: {n: v.tolist() for n, v in p.items()} for c, p in prototypes.items()},
                "gaussian_pooled_training_std": pooled_std.tolist(),
                "gaussian_std": {c: s.tolist() for c, s in gaussian_std.items()},
                "logistic_parameters": LOGISTIC, "logistic_iterations": logistic[-1].n_iter_.tolist(),
                "fit_warnings": [str(w.message) for w in caught],
                "variance_support": logistic[0].get_support().tolist(),
                "scaler_mean": logistic[1].mean_.tolist(), "scaler_scale": logistic[1].scale_.tolist(),
                "classes": logistic[-1].classes_.tolist(), "coef": logistic[-1].coef_.tolist(),
                "intercept": logistic[-1].intercept_.tolist()}
    for c, p in profiles.items():
        raw = raw_by_composer[c]
        metadata["profiles"][c] = {"fingerprint": p.fingerprint, "train_sample_ids": p.train_sample_ids,
            "feature_names": p.feature_names, "feature_groups": p.feature_groups, "mean": p.mean.tolist(),
            "std": p.std.tolist(), "raw_std": raw.std(axis=0).tolist(),
            "std_replaced_indices": np.flatnonzero(raw.std(axis=0) < 1e-6).tolist()}
    return {"profiles": profiles, "gaussian_std": gaussian_std, "prototypes": prototypes, "logistic": logistic}, metadata


def score_measures(fitted: dict, vector: np.ndarray, custom: np.ndarray, events: dict) -> dict:
    scores = {m: {} for m in MEASURES}
    for c, p in fitted["profiles"].items():
        scores["rms67"][c] = -float(np.mean(list(grouped_distance(vector, p).values())))
        satisfaction = np.exp(-0.5 * np.square((vector - p.mean) / fitted["gaussian_std"][c]))
        scores["gaussian67"][c] = float(np.mean([satisfaction[np.asarray(p.feature_groups) == g].mean() for g in GROUPS]))
        for name in ("onset_duration", "time_pitch"):
            scores[name][c] = cosine(events[name], fitted["prototypes"][c][name])
    probabilities = fitted["logistic"].predict_proba(np.asarray(custom).reshape(1, -1))[0]
    scores["logistic93"] = dict(zip(fitted["logistic"].classes_, map(float, probabilities)))
    for measure, values in scores.items():
        if any(v is not None and not np.isfinite(v) for v in values.values()):
            raise ValueError(measure + " nonfinite affinity")
    return scores


def rank_credit(scores: dict, composer: str) -> tuple[float | None, list[str]]:
    if any(v is None or not np.isfinite(v) for v in scores.values()) or not scores:
        return None, []
    maximum = max(scores.values())
    tied = sorted(c for c, v in scores.items() if abs(v - maximum) <= 1e-12)
    return (1 / len(tied) if composer in tied else 0.0), tied
