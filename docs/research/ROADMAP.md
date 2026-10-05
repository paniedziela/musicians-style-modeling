# ROADMAP — revised Research V2 implementation boundary

Current checkpoint (2026-10-06): Track A external evaluation is complete after reviewed A+B commits and clean-branch push. jSymbolic work-balanced attribution 0.952381 versus custom93 0.783069; 457/638 external dimensions retained; 138/138 extraction successes; 9/18 output sign agreement. Exact external bundle provisionally frozen for attribution diagnostics, with substantial transformed-feature extrapolation and no musical-validity claim. See [A completion](AB_A_COMPLETION.md). No B execution, outer-test source/features, tuning, main change, merge or rebase. Stop for review after A. Historical entries below retain their original boundaries.

Updated 2026-10-05. V2-05 fixed-budget objective pilot is complete: 18/18 runs completed; execution/preservation passed=True. Stop after V2-05 for review; no full E1c, outer-test evaluation, larger transfer experiment or push. V2-01 through V2-04 remain preserved. See [V2_05_COMPLETION.md](V2_05_COMPLETION.md) and [V2_05_PROTOCOL.md](V2_05_PROTOCOL.md).

## Scope and architecture

Keep existing repository layout, E1–E4 source, configurations, feature contracts, schemas and artifacts stable. New modules may wrap/import frozen implementations; frozen experiments must not depend on them. Introduce reusable capabilities only for concrete needs, with small modules first. V2-01 adds root/provenance/audit support; `content_metrics` was introduced for V2-03 and `feature_backends` only for V2-02; `style_metrics.py` and the explicit `style_audit.py` runner were introduced only for concrete V2-04 needs; no registry/framework was added. No tagging, worktree creation, branch changes or speculative search refactor is required for this pass.

Root configuration order: explicit options, then MSM_DATA_ROOT / MSM_RESULTS_ROOT / MSM_LITERATURE_ROOT, then active checkout's datasets / experiments / Literatura. Resolve relative options against that checkout and record selection sources. Do not relocate/duplicate local assets or change legacy loaders. Separate source checkout from asset roots when using another worktree.

## Test strategy

| Tier | Selection / purpose |
|---|---|
| Fast unit/smoke | Default `python -m pytest`: tests/unit, checkout-first source, excluding property/integration/regression/slow |
| Property | Explicit tests/property; development profile caps every original decorator at 20; full retains original 50/100/200/300 budgets |
| Integration/regression | Explicit component, live HTTP, training, checkpoint-to-MIDI and costly plotting/report/export checks; retained assertions and node locations |
| Scientific audits | Explicit read-only filesystem/hash/content audits with fresh artifacts; never ordinary dataset-wide pytest cases |

The previous 493-unit run took 50.46 s. Timing evidence identified costly inference, checkpoint, plotting and retraining workflows. Tier markers remove them from default selection without deleting tests. Inexpensive model-shape/loss checks remain fast. Target default below 30 s here, without a brittle timing assertion. The observed first revised run is 452 passed / 64 deselected in 19.59 s; development property run is 17 passed in 9.34 s. Static test inventory counts functions, not parametrized pytest cases.

Commands from the active checkout with its environment activated:

```powershell
python -m pytest
python -m pytest -o addopts= tests/property --hypothesis-profile=dev
python -m pytest -o addopts= tests/property --hypothesis-profile=full
python -m pytest -o addopts= tests/unit tests/integration -m "integration or regression"
python -m pytest -o addopts= tests/unit tests/integration
python -m pytest -o addopts= tests --hypothesis-profile=full
python tools/research_audit.py --scope inventory --output experiments/<fresh-inventory-directory>
python tools/research_audit.py --scope baseline --output experiments/<fresh-baseline-directory>
```

An explicitly selected property suite defaults to full unless dev is requested. `-o addopts=` removes the default exclusions. An unfiltered full run includes the slow properties; it is not the default iteration command. 150-source / 600-output provenance and later 600-output content checks write scientific audit artifacts instead of entering pytest. Preserve valuable counterexample strategies/assertions/deadline policy. Profile-aware decorators are necessary because loading a profile alone cannot override explicit max_examples.

## Bounded backlog

### V2-01 — roots, environment provenance, evidence audit and test tiers — MUST

**Reason:** avoid stale installed imports, false worktree provenance and mixing expensive scientific checks with development tests.

**Read:** STATUS/master prompt; frozen E1 loaders/manifest/splits; E2/E3 results/run manifests/snapshots; E4.6 status/validation/checkpoint; package config, README, pytest and property settings; local selected PDFs and historical reports.

**Allowed files:** `src/musicians_style/asset_paths.py`, `provenance.py`, `research_audit.py`; thin `tools/research_audit.py`; synthetic unit/subprocess tests; five research docs and README; pytest markers/configuration/property-settings support. Do not modify frozen loaders or experiments.

**Import contract:** inspect ambient musicians_style.__file__ before loading research modules; resolve strictly under the active checkout's `src/musicians_style` and check package search locations. Merely being inside the checkout is insufficient: reject `.venv` copies and other worktrees. Never silently repair. On failure reserve a fresh diagnostic directory, record failure/provenance, explain editable or checkout-first setup and exit nonzero. Prefer editable `pip install --no-deps --no-build-isolation -e .`; checkout-first PYTHONPATH is the fallback, with resolved source recorded.

**Provenance/artifacts:** checkout/expected package/actual source/search locations; executable/Python/platform/dependencies/editable metadata; HEAD/branch/worktree/dirty state; absolute roots/selection sources/config fingerprints; `provenance.json`, `asset_inventory.json`, `audit.json`, `test_inventory.json`, readable report in a new output directory. Existing destinations are rejected. `inventory` remains lightweight; `baseline` explicitly verifies manifest counts/source MIDI hashes/XML, sample/group/hash-disjoint frozen splits, canonical snapshots/recorded inputs and E2/E3 outputs, E4.6 checkpoint/validation availability and historical provenance limitations. No fitting, extraction, metrics or regenerated MIDI.

**Acceptance:** synthetic roots/config/env/default/alternate-worktree cases; correct source import, stale-in-checkout/foreign/missing/unresolved imports; real subprocess guard before research imports; missing asset/hash mismatch/conflicting snapshot/split/checkpoint diagnostics; existing-output rejection; byte-identical synthetic frozen inputs. Real baseline checks are explicit artifacts. Verify fast tests, dev properties, full-budget retention, broad existing unit/integration/regression coverage and final source/config/evidence fingerprints.

**Estimate:** MEDIUM; 3–4 agent passes (implementation, tier organization, verification, review cleanup); no training, measured tests plus filesystem/hash scans. No blocking dependency. Read-only audits may run independently; edits and environment changes remain sequential.

**Non-goals:** feature-backend/metric APIs, dataset extraction, content measurements, transfer/training, historical provenance repair. **STOP for review after V2-01.**

### V2-02 — feature feasibility — COMPLETE, stop for review

**Reason:** test whether established MIR extractors can add useful reliable signal before E1c. **Read:** frozen composition_features/manifest/splits, E1b schema, matrix L0034/L0230, musif docs/version constraints. **Implemented:** capability-oriented `src/musicians_style/feature_backends` custom93/musif adapters, explicit feasibility runner, isolated dependency lock, synthetic tests and new artifacts; never frozen E1 code.

Wrap custom93 without duplication. Select nine training samples: first three lexical work groups per composer in repeat 0 / outer fold 0 train, one lexical sample each; persist identities/hashes before extraction. Isolate musif dependencies. **Acceptance:** all nine samples have features or explicit failure records on both attempts (18 extraction attempts); compare complete pilot schemas, values or failure statuses for determinism. Record empty/nonfinite/missing values, sample mapping, versions and runtime; no hidden sample exclusion. Dataset-wide custom93/cache comparison produces an audit artifact. No E1c model fitting in this bounded pilot.

**Artifacts:** pilot manifest, extractor schema/cache/failure records, determinism comparison and feasibility report. **Estimate:** MEDIUM; 2–3 passes; 18 pilot attempts per backend plus explicit cache audit, no training. Requires V2-01 and isolated MIR dependencies. May run alongside V2-03 after separate review. Stop before full E1c or another backend.

**Completion:** nine predeclared training samples; two attempts per backend (36 total), no dropped samples. Exact repeated results; all 150 custom93 cache vectors match. musif MIDI produces 203-329 numeric nullable features per sample, 405 in the descriptive union; missing tempo and variable vocabulary are limitations for subsequent model use. Exact evidence: [V2_02_COMPLETION.md](V2_02_COMPLETION.md), `experiments/research_v2_02_2026-10-04/pilot_final`. Run `.venv/Scripts/python.exe -B tools/feature_feasibility.py --output experiments/<fresh-pilot-directory> --musif-python experiments/research_v2_02_2026-10-04/musif_env/Scripts/python.exe`. No full E1c, another backend, V2-04 or V2-05 launch. **STOP for V2-02 review.**

### V2-03 — content audit — COMPLETE, artifacts preserved

**Reason:** quantify preservation independently from style before objective comparison. **Read:** frozen MIDI/event handling and E3 melody/transposition constraints; E2/E3 outputs; L0074/L0185/L0208. **Implemented:** concrete `src/musicians_style/content_metrics.py`, `content_audit.py`, thin `tools/content_audit.py`, focused synthetic tests and new artifacts. No reverse dependencies or frozen source/configuration/result/report changes.

**Acceptance:** protected original pitch/onset/duration/note-off/order exact for new policy; structural meter/metadata/format/resolution/channel checks separate; velocity reported as transformation invariant, not assumed fundamental melody content. Test identity/corruption, overlaps and event-order ambiguity, historical relative/transposed protection and strict new policy. Audit all 600 frozen outputs explicitly, with per-output success/failure and clustered direction summaries. Preserve E3's original interpretation and all inputs.

**Artifacts:** metric contract/version, per-output audit/ambiguity records and report. **Estimate:** MEDIUM; 2–3 passes; parsing/measurement of 600 outputs, no generation. Requires V2-01; may run alongside V2-02. Non-goals: optimizer changes, synthetic regeneration, objective selection.

**Completion:** all 600 frozen outputs plus 150 original identity references audited explicitly. Final files, commands/results, findings, ambiguity limits and artifact locations: [V2_03_COMPLETION.md](V2_03_COMPLETION.md). Reproduce with `.venv/Scripts/python.exe -B tools/content_audit.py --output experiments/<fresh-content-audit-directory>`; roots and ambient import guard follow V2-01. The fast/property/regression tiers are unchanged. No V2-02 work ran alongside this audit. **The V2-03 stop boundary preceded the separately authorized V2-02 task.**

Review clarification (recorded before V2-02 implementation): distinct simultaneous same-tick order is diagnostic, not a hard melodic-identity violation on its own. Exact protected pitch, onset and note-off / identifiable duration remain primary. The completed V2-03 strict-order contract/statistics and artifacts above are retained unchanged.

### V2-04 — E1d style-measure comparison — COMPLETE, stop for review

Implemented concrete `src/musicians_style/style_metrics.py`, `style_audit.py`, thin `tools/style_audit.py`, synthetic checks and definitive artifacts. Frozen code has no reverse imports. [V2_04_PROTOCOL.md](V2_04_PROTOCOL.md) fixes formulas/fitting/ties/uncertainty before scores; [V2_04_COMPLETION.md](V2_04_COMPLETION.md) records exact verification/provenance/deviations. No broad replanning, musif requirement, E1c selection or transfer search.

Gaussian67 uses the recorded V2 train-only floor `max(target std, 0.05 * pooled training std, 1e-6)` and equal E3 family means of `exp(-z²/2)`, not E3's near-zero std replacement. Logistic93 uses fixed C=1, balanced, train-only variance/scaling, max_iter=5000; lbfgs is an implementation choice, not a previously fixed solver. RMS imports frozen behavior; the two event-profile cosines stay separate.

25 outer-training bundles; 750 real held-out observations from 150 pieces/87 works; 300 exactly task-aligned E2/E3 pairs and 300 target-directed source-self references. All 150 source custom93 vectors exactly match the frozen cache. All style scores are defined/nondegenerate, all source-self deltas zero. Definitive audit passed in 348.13 s; all 3,573 protected files preserved. Artifacts: `experiments/research_v2_04_2026-10-05/style_verified`.

| Measure | Work-balanced ranking | Clustered 95% CI | E3 positive mean directions | E3 negative / null tasks |
|---|---:|---|---:|---|
| rms67 | 0.554152 | [0.500890, 0.610256] | 6/6 | 10/300; 32/300 |
| gaussian67 | 0.558248 | [0.508220, 0.610659] | 6/6 | 17/300; 32/300 |
| logistic93 | 0.823703 | [0.768912, 0.877364] | 6/6 | 57/300; 45/300 |
| onset_duration | 0.397947 | [0.337051, 0.456813] | 5/6 | 123/300; 46/300 |
| time_pitch | 0.697950 | [0.629007, 0.766048] | 4/6 | 66/300; 92/300 |

All five ranking CI lower bounds clear 1/3, but onset-duration is marginal (0.337051). Logistic93 and time-pitch have the strongest held-out ranking here. E3 positive mean directions are 6/6 for RMS/Gaussian/logistic, 5/6 for onset-duration and 4/6 for time-pitch; several direction CIs cross zero. E2 has negative means in all six directions under RMS/Gaussian/event profiles, but logistic has five positive means. These definitions disagree materially; V2-04 did not select an objective. The user subsequently fixed the three V2-05 objectives.

Saved-MIDI RMS agrees with historical E3 gain within 1e-10 for 287/300 outputs. Thirteen serialized identity fallbacks across seven works share V2-03 Skyline reselection discrepancies (max difference 0.002693). These remain explicit diagnostics; no selector/output/delta is repaired and no threshold is tuned. Source-self identity and serialized-fallback null rates remain distinct.

RMS/Gaussian share the 67-component representation, but their predeclared variance policies differ: link function and variance handling are not isolated separately. Logistic93/historical RF share corpus/custom93 and are separate held-out evidence, not fully independent. Event profiles are all-piano adaptations using fixed four-quarter-beat windows; 117/150 sources contain non-4/4 meter and 65 have meter changes. Time-pitch ignores global transposition/duration; onset-duration has weak ranking and frequent negative E3 movement. No perceptual validity or fresh untouched confirmatory test is claimed.

Review ranking/clustered uncertainty, spread/missingness, determinism, direction consistency, null/negative rates, agreement structure, leakage and interpretability together. No single scalar promotes a candidate; failed candidates remain diagnostic evidence. **V2-04 review boundary was satisfied by the explicit bounded V2-05 authorization; full E1c remains unstarted.**

Reproduce: `.venv/Scripts/python.exe -B tools/style_audit.py --output experiments/<fresh-style-audit-directory>`. V2-04 did not select a future objective; the user fixed the V2-05 set after review.

### V2-05 — fixed-budget objective pilot — COMPLETE, stop for review

User-fixed RMS67/Gaussian67/logistic93, lexical r00/f00/inner00 validation cohort, all six directions, 18 runs. Common seed1729/population32/generations60/stagnation61, inherited E3 elite/tournament/sigmas; external zero-transpose canonicalization preserves frozen GA code/draws. One inner-training-only fit; time_pitch primary distinct evaluation, onset_duration secondary. No larger experiment or outer-test evaluation. Exact findings/provenance/tests/limitations: [V2_05_COMPLETION.md](V2_05_COMPLETION.md), fixed contract [V2_05_PROTOCOL.md](V2_05_PROTOCOL.md).

## Later optional work

Full E1c and matched MusicXML E1m follow feasibility review. StyleRank/relative/sequential objectives and a transposition-policy ablation are later bounded proposals. Preserve E4.6 NO-GO; pretrained neural comparisons need separate feasibility/evaluation evidence. No automatic progression through this backlog.

## V2-05 exploratory result

| Objective | Exact observable policy | time_pitch positive / negative / null | time_pitch mean | Optimized mean |
|---|---:|---|---:|---:|
| rms67 | 6/6 | 1 / 5 / 0 | -0.022067 | 0.377232 |
| gaussian67 | 6/6 | 2 / 4 / 0 | -0.026364 | 0.026629 |
| logistic93 | 6/6 | 0 / 6 / 0 | -0.068912 | 0.139354 |

Definitive artifacts: `experiments/research_v2_05_pilot_2026-10-05`. All comparisons are exploratory, with content ambiguity/evaluator dependence recorded separately. Optimized gains use different scales and cannot rank objectives across metrics. Stop for V2-05 review.
