# ROADMAP — revised Research V2 implementation boundary

Updated 2026-10-04. **V2-01, V2-03 and the separately authorized V2-02 feature feasibility are complete. Stop for review after V2-02; local commits authorized, no push. Do not start full E1c, V2-04 or V2-05.** Research V2 is an experiment phase, not `musicians_style.v2` or a second implementation tree. The remaining backlog is unchanged; the V2-01 boundary below is historical.

## Scope and architecture

Keep existing repository layout, E1–E4 source, configurations, feature contracts, schemas and artifacts stable. New modules may wrap/import frozen implementations; frozen experiments must not depend on them. Introduce reusable capabilities only for concrete needs, with small modules first. V2-01 adds root/provenance/audit support; `content_metrics` was introduced for V2-03 and `feature_backends` only for V2-02; `style_metrics` remains a future task location, not foundation scaffolding. No tagging, worktree creation, branch changes or speculative search refactor is required for this pass.

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

### V2-04 — E1d style-measure comparison — MUST, planning only

**Reason:** weak E3 agreement requires measurement validity before search compute. **Read:** E1 grouped fitting/evaluator, E3 67-component RMS implementation, V2-03 content audit and matrix L0034/L0067/L0074/L0230. **Allowed later:** concrete `src/musicians_style/style_metrics` and E1d runner/config, small tests and new artifacts; adapters can import frozen code.

**Acceptance:** train-only fitting for named candidate RMS/Gaussian/classifier/event-profile measures; identities, real held-out works and frozen E2/E3 scored; finite/nondegenerate scores, deterministic behavior, leakage tests and grouped clustered target-ranking assessment. Explain that RF is a separate held-out evaluator with corpus/feature dependence; event profiles provide a structurally distinct perspective. Predeclare tie handling, clustering/CI and above-chance eligibility before scores. Do not average every score or tune selection on transfer outcomes. Failed measures remain diagnostic evidence.

**Artifacts:** fit/scoring provenance, fold predictions, ranking/agreement/content-dependence report and objective eligibility table. **Estimate:** MEDIUM; 3–4 passes; fold-scoped CPU fitting/cached scoring, no transfer search; requires V2-01/V2-03, optional V2-02 features. Extraction caching may proceed independently. Stop for review and candidate selection.

### V2-05 — objective pilot — SHOULD, provisional and planning only

**Reason:** test eligible E1d objectives under exact protected melody and equal bounded compute. **Read:** completed E1d/eligibility and content contract, frozen E3 engine/feasibility, baseline artifacts. **Allowed later:** external adapter/runner/config and tests/artifacts. Prefer zero-transposition enforcement outside frozen search. No duplicated engine. E3 source changes require demonstrated adapter insufficiency, separate minimal-change review, unchanged old default and regression tests.

**Acceptance:** after E1d and another review choose up to two eligible candidates; no fixed RMS67/Gaussian67/classifier93 pilot list. Require leakage-safe fit, finite/nondegenerate/deterministic scores and grouped held-out target-ranking clustered CI lower bound above chance (1/3). If none passes, NO-GO without transfer optimization. Predeclare source works, seeds, six directions, budget and exact-pitch/onset/note-off/identifiable-duration constraints, with distinct same-tick event ordering diagnostic; reject/log nonzero transposition. Report identity/frozen paired baselines, content/style separately, evaluator dependence and failures.

**Artifacts:** reviewed candidate selection and protocol, measured pilot budget, new outputs/reports under new IDs. **Estimate:** complexity/passes finalized after E1d; six-direction cost measured before launch; requires V2-04 completion and review, cannot parallelize with candidate selection. Non-goals: automatic ablation, shared E3 refactor or neural tuning.

## Later optional work

Full E1c and matched MusicXML E1m follow feasibility review. StyleRank/relative/sequential objectives and a transposition-policy ablation are later bounded proposals. Preserve E4.6 NO-GO; pretrained neural comparisons need separate feasibility/evaluation evidence. No automatic progression through this backlog.
