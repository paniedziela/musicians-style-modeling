# V2-04 completion record — 2026-10-05

V2-04 E1d style-measure audit only is complete. Stop for review. No future V2-05 objective selected, full E1c classification/model selection, StyleRank, sequential/neural work, generator, transfer optimization, output regeneration or push. All findings below use only `style_verified`.

## Exact changed files (11)

Paths relative to `D:/Studia/inzynierka_dev`:

- `src/musicians_style/style_metrics.py`: concrete train-only RMS67/Gaussian67/logistic93 and separate onset-duration/time-pitch cosines; leakage and explicit score/tie contracts. Imports frozen E3/custom93 functionality; no copied feature implementation or registry.
- `src/musicians_style/style_audit.py`: explicit frozen-input/split/task validation, extraction/cache equivalence, persisted fitting before evaluation, separate real/output/identity scoring, clustered ranking/movement/correlation summaries, historical/serialization and content-dependence diagnostics, fresh artifacts and preservation checks.
- `tools/style_audit.py`: thin guarded entry point, checkout import verification before research-module loading and fresh destination.
- `tests/unit/test_style_metrics.py`: leakage, fixed Gaussian floor, frozen RMS equality, train-only poisoning/cached representation validation, deterministic logistic fitting, event bins/overflow/symmetry/transposition/empties/ties, and duration/velocity reassociation counterexample.
- `tests/unit/test_style_audit.py`: exact task/split alignment, sample/group/hash and transformed-hash rejection, grouped denominators/undefined correlations, direction/null semantics, import/protected-destination guards, full synthetic read-only audit and explicit source/hash/snapshot/cache/missing failures; historical disagreement stays diagnostic.
- `docs/research/V2_04_PROTOCOL.md`: pre-scoring formulas, cohort/fitting provenance, ties/null conventions, bootstrap/agreement and multi-evidence review contract.
- `docs/STATUS.md`, `docs/research/ROADMAP.md`, `docs/research/RESEARCH_CONTEXT.md`, `docs/research/EXPERIMENT_REGISTRY.md`: bounded completion/findings/limitations and stop boundary; remaining backlog unchanged.
- `docs/research/V2_04_COMPLETION.md`: this completion record.

35 new synthetic cases: 29 fast and six explicitly integration-marked. Existing test configuration, tiers/budgets and tests are unchanged. Scientific artifacts remain ignored by existing Git policy. No speculative interface, framework, parallel namespace or reverse import from frozen E1–E4.

## Fixed protocol and fitted provenance

Canonical 150 samples / 87 works: Bach 59/30, Beethoven 57/28, Chopin 34/29. All 25 frozen outer folds and 75 inner partitions are validated for coverage/duplicates and sample/group/SHA disjointness before fitting. Inner folds are unused: no selection. Every transformed held-out output is also checked against original training hashes. All 150 re-extracted custom93 vectors exactly match frozen e1.3.0 names/order/units/values and source hashes; snapshots must agree.

25 fold bundles were persisted before the first held-out score: 25 logistic fits plus 75 composer RMS profiles, Gaussian scales and pairs of event prototypes. All 25 serialized bundle scores exactly reproduce in-memory scores. `fits/r00_f00` through `fits/r04_f04` contain joblib bundles and JSON with all train/forbidden sample/group/hash identities, means/raw/effective std, feature names/groups, Gaussian pooled std, event prototypes, variance support, scaler mean/scale, classes/coefficients/intercepts/iterations/warnings and serialized hashes. `fit_manifest.json` fingerprints both model and metadata files. There are **zero fitting/convergence warnings**.

- RMS67: negative equal-family mean of frozen E3 RMS z distances, with frozen `std < 1e-6 -> 1.0` behavior.
- Gaussian67: recorded V2 adaptation `exp(-z²/2)`, mean within each E3 family then equal family mean; train-only `max(target-composer std, 0.05 * pooled-training std, 1e-6)`. Pooled includes all outer-training composers. This is bounded marginal criterion satisfaction, not density/likelihood or frozen E3 variance behavior.
- Logistic93: recorded C=1, balanced classes, train-only variance filtering/scaling, max_iter=5000; target probability affinity. lbfgs/L2, tol=1e-4 and seed 1729 are implementation choices/defaults; the prior V2 record did not fix solver. No model selection/calibration.
- Event profiles: all parsed piano notes; separate 24×12 onset-duration and 24×41 forward-lag/signed-pitch-pair profiles, cosine to equal-work train prototypes. Four-quarter-beat windows in every meter; >=2-beat durations clipped, pitch intervals outside -20..20 excluded/count recorded, lag >=4 excluded, simultaneous pairs symmetrized. Empty query/prototype is undefined. This borrows audited Groove2Groove principles without claiming exact upstream/BIAB reproduction.

The Gaussian settings were supplied explicitly by the user when repository/original local bundle docs lacked the numbers. The proposed E3 fallback was **never used for scored Gaussian results**. The recorded numerical contract appears in every run's protocol.json before fitting/scoring. No musif import/extraction/installation or dependency change was needed.

## Held-out real works and transfer findings

750 held-out piece observations (150 unique pieces × five repeats), averaged within 87 original work groups, then within composer, then macro composers. Ranking ties within 1e-12 receive fractional 1/k credit. Stratified work bootstrap: 2,000 draws, seed 1729, percentile 95% intervals; chance 1/3. Rows/defined/undefined/work counts and composer recalls/true-versus-counterexample margins are explicit in summary.json. No score slot or ranking row is missing, no real ranking ties, and all five measures are nondegenerate.

| Measure | Work-balanced ranking | Clustered 95% CI | E3 negative / null | Positive E3 mean directions |
|---|---:|---|---|---:|
| rms67 | 0.554152 | [0.500890, 0.610256] | 10/300; 32/300 | 6/6 |
| gaussian67 | 0.558248 | [0.508220, 0.610659] | 17/300; 32/300 | 6/6 |
| logistic93 | 0.823703 | [0.768912, 0.877364] | 57/300; 45/300 | 6/6 |
| onset_duration | 0.397947 | [0.337051, 0.456813] | 123/300; 46/300 | 5/6 |
| time_pitch | 0.697950 | [0.629007, 0.766048] | 66/300; 92/300 | 4/6 |

Logistic93 and time-pitch have the strongest real-work ranking. RMS/Gaussian show moderate corpus discrimination. Onset-duration only narrowly clears chance (CI lower bound 0.337051), has 123/300 negative E3 tasks and uncertain directional effects. **Ranking alone does not promote any candidate.** Review score behavior, leakage, direction consistency, agreement and interpretation together; all measures remain recorded and no future pilot set was selected.

All 300 pairs align by exact task ID, source/group/composer/target/fold/seed/phase, with matching frozen E3 training IDs/profile fingerprints. Each direction has 59 Bach-source tasks/30 works, 57 Beethoven-source tasks/28 works or 34 Chopin-source tasks/29 works. Both E2/E3 use matching repeat-0 models. The 300 target-directed identity references use 150 original sources; they are not 300 distinct works. Every identity delta is exactly zero (0/300 negative; 300/300 null) under every measure.

| Measure | E2 target movement, equal-work mean | E3 target movement, equal-work mean | E2 negative / null |
|---|---|---|---|
| rms67 | -0.252821 [-0.300140, -0.212965] | 0.518540 [0.345229, 0.714111] | 263/300; 0/300 |
| gaussian67 | -0.035028 [-0.039587, -0.030468] | 0.027136 [0.022730, 0.032026] | 255/300; 0/300 |
| logistic93 | 0.086286 [0.045607, 0.127751] | 0.060035 [0.035308, 0.088950] | 138/300; 0/300 |
| onset_duration | -0.262175 [-0.296935, -0.227125] | 0.011645 [-0.001235, 0.024345] | 272/300; 0/300 |
| time_pitch | -0.188767 [-0.207875, -0.168013] | 0.003305 [-0.001711, 0.009265] | 258/300; 2/300 |

Pooled movement intervals resample the 87 original works, averaging tasks within work with equal work weights. All 300 output-task deltas per experiment/measure are defined. The readable report and summary.json include **all six directions separately**, target ranking/margin movement, source drop, negative/null rates and their clustered intervals. The measures have different units; magnitude across measures is not directly comparable.

E3 mean direction signs: 6/6 positive RMS/Gaussian/logistic, 5/6 onset-duration, 4/6 time-pitch. Some CIs cross zero: e.g. Bach→Beethoven logistic [-0.032751, 0.044629], onset-duration [-0.010346, 0.007053]. E2 has 0/6 positive means under RMS/Gaussian/event profiles, while logistic has 5/6. Paired E3-minus-E2 logistic movement is -0.026251 [-0.074507, 0.024971], so this measure does not establish an E3 advantage. Historical E3 has 45/300 identity genomes; byte-identical output/source files are 0/300 for both frozen experiments, reported separately from semantic/metric nulls.

## Agreement, serialization and content separation

All ten style-measure pairs have Pearson/Spearman output-level and work-mean coefficients with clustered intervals, pooled/per direction and separate identity cases. Real true-composer affinity and candidate-composer-specific agreements are also recorded. Constant/<3 work-pair cases and degenerate bootstrap draws are explicit undefined records, not zero correlations. Historical E3 objective comparisons are defined only for E3; E2/identity cases remain undefined.

For E3 RMS versus Gaussian movement: output-level Pearson 0.283849, Spearman 0.777267; work-mean Pearson 0.415565, Spearman 0.653245. Holding the 67 features fixed still yields distinct score behavior. The predeclared scales differ too, so this comparison does not isolate the link function separately from variance handling.

RMS versus historical E3 internal gain: output Pearson 0.999999983, Spearman 0.998853; work Pearson 0.999999939, Spearman 0.999499. 287/300 agree within 1e-10; **13 identity-fallback outputs across seven works differ**, max 0.002692971819. All 13 also have V2-03 Skyline reselection discrepancies. The frozen FIFO parser/serialization can reassociate durations to velocities at identical attacks; the velocity/duration-sensitive Skyline tie can change which duration is excluded from accompaniment. The synthetic counterexample verifies this mechanism. V2-03 protected FIFO equality is not equality of every note/velocity tuple. No frozen selector, output or measured delta was repaired. This explains why RMS/Gaussian null counts are 32 rather than the 45 historical identity genomes; all 10 negative RMS deltas occur in these cases.

Agreement with historical E3 gain for logistic93 is weak at output level (Pearson 0.021256, Spearman 0.212961). Time-pitch output Pearson is -0.137900, Spearman 0.089202. Historical RF deltas are a **separate held-out evaluator**, with shared-corpus/custom93 dependence. RMS/RF output Pearson 0.043957 closely tracks the historical weak agreement; work averaging changes it to 0.411456. Both estimands/denominators are retained rather than selecting the flattering one. Logistic is a separate style measure and shares corpus/features with RF; neither supplies fully independent evidence.

V2-03 content contract/results are read/hash-joined, not recomputed or combined into style. Exact-pitch onset retention correlations have 283/300 defined E2 tasks (17 raw-event undefined cases), and 300/300 E3 tasks. Style scores remain defined on the frozen parsed representation; that does not resolve raw MIDI ambiguity or prove protected content retention. Identity/content/style axes stay separate.

## Diagnostics and limitations

Every measure has 4,950 defined affinity slots across 750 real rows and 900 identity/output rows, zero missing slots; detailed ranges/std/exact unique counts and per-population degeneracy are in summary.json. Empty profiles are explicit in code/tests but none occur here. Identity movement correlations are undefined because all deltas are constant zero. No undefined correlation is silently dropped.

117/150 original sources have some non-4/4 meter; 65 have meter changes. Source onset-duration profiles clip 9,311/365,680 notes at >=2 beats and diagnose 10,206 nonpositive durations. Time-pitch excludes 2,112,621/15,073,453 source note-pair observations for pitch overflow. E2/E3 corresponding counts remain separate in event_diagnostics. Those choices are predeclared adaptations, not fitted corrections to transformed outputs. Time-pitch intentionally ignores global pitch transposition and durations; null movement can be appropriate for changes outside that representation. Frozen E3's historical ±6-semitone policy remains distinct from the future exact-pitch contract.

Correlated bins, family weighting, sparse composer variance, Skyline selection, corpus/meter mixture and FIFO representation affect interpretation. E3 was optimized under RMS already, so positive E3 RMS gain is selection-dependent evidence. Work bootstrap is descriptive/conditional on fitted folds, does not refit training models and has no multiplicity correction. Repeated held-out observations and repeated development audits are not independent samples. Frozen folds were inspected historically; this is not a new untouched confirmatory or listening test. Corpus discrimination is not universal/perceptual composer style validity. Historical E1–E4 provenance gaps remain unrepaired.

## Exact verification commands/results and runtime

Working directory `D:/Studia/inzynierka_dev`; original `.venv/Scripts/python.exe`, Python 3.10.20, numpy 1.26.4, scipy 1.13.1, sklearn 1.5.1. No installs/dependency upgrades. -B prevents bytecode writes; cacheprovider is disabled for the existing inaccessible pytest cache.

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --durations=10` | **549 passed / 70 deselected**, 14 existing warnings; **32.57 s pytest / 34.57 s command wall** |
| `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider -o addopts= tests/unit/test_style_metrics.py tests/unit/test_style_audit.py tests/unit/e3 tests/unit/e1/test_composition_features.py tests/unit/test_midi_parser.py tests/unit/test_v2_content_metrics.py tests/unit/test_asset_paths.py tests/unit/test_provenance.py tests/unit/test_research_audit.py --durations=10` | **136 passed**, 14 existing warnings; **23.00 s pytest / 24.26 s command wall** |
| `.venv/Scripts/python.exe -B tools/research_audit.py --scope baseline --output experiments/research_v2_04_2026-10-05/baseline_before` | Passed, **24 passed / 2 unavailable / 0 failed**, 8.25 s command wall |
| `.venv/Scripts/python.exe -B tools/research_audit.py --scope baseline --output experiments/research_v2_04_2026-10-05/baseline_after` | Same results, 4.90 s subprocess wall |
| `.venv/Scripts/python.exe -B tools/style_audit.py --output experiments/research_v2_04_2026-10-05/style_verified` | **Passed**, 750 real / 900 output+identity rows, **348.13 s** audit phase, excluding final artifact serialization |
| AST parsing/new-source checks, frozen E1–E4 reverse-import scan, `git diff --check` and staged check | Passed |

Final fast/focused suites ran alongside the definitive audit; timing is measured under that load and no test tier was changed to force a target duration. Logs and structured command results are `fast_verified.*`, `focused_verified.*`, `baseline_after_command.json`. Earlier fast/focused results (548/134 cases) remain in `fast_tests.*` / `focused_tests.*`; new regression diagnostics justified final reruns. No broad expensive suite/full-budget property rerun was needed for unchanged frozen code.

The same style command also ran with `style_initial` (730.68 s) and `style_final` (427.28 s). Both scored all data and passed coverage/leakage/preservation/serialization checks, but reported execution failure solely because an unintended exact historical-RMS agreement gate treated the 13 genuine serialization diagnostics as integrity failures. The correction retains the fixed 1e-10 discrepancy marker and every measured difference, and removes that execution gate. **No score formula, variance floor, model configuration, metric/null threshold or measured value changed.** All 25 model bundles and all 750 real/900 movement-score records are identical between development and definitive runs. Added diagnostics and provenance/reporting differ. Development artifacts remain inspectable; only style_verified is definitive.

## Preservation, artifacts and local commits

Zero changed/missing/added files among **3,573** before/after hashes: frozen E1–E4 complete run trees/source/configs and reports, reused project code/tests, requirements/configuration, canonical JSONs/150 source MIDIs, all completed V2-01/02/03 scientific artifacts and completed V2-02/03 records/code. The isolated musif environment directory and bytecode are excluded from this artifact fingerprint; it was neither used nor modified. Original project environment versions/import are identical before/after. Separate baseline fingerprints match for **1,004 evidence files, 34 frozen experiment sources, nine configurations and 13 selected papers**. Post-run verification rechecks these and all definitive source/model/artifact hashes. Frozen E1–E4 never import the new style code.

Definitive absolute directory: `D:/Studia/inzynierka_dev/experiments/research_v2_04_2026-10-05/style_verified`:

- `protocol.json`, `provenance.json`: fixed metric/uncertainty contract, checkout/import/environment/Git/roots and actual source hashes; scoring-start time follows all saved fits.
- `fits/*.json`, `fits/*.joblib`, `fit_manifest.json`: all 25 train-only bundles and learned provenance.
- `feature_cache.json`: all 150 source/600 output representations and event diagnostics, with paths/hashes; no new original assets.
- `real_scores.json`: all 750 held-out rows, all candidate-composer affinities/ranks.
- `alignment.json`, `per_output.json`: 300 exact pair records and 900 separate identity/E2/E3 score/movement/content-diagnostic rows.
- `summary.json`: ranking/composer/direction results, score degeneracy, clustered rates/uncertainty, correlations/undefined cases, paired E3-minus-E2, serialization rows and review evidence table.
- `audit.json`, `frozen_sha256.json`, `report.md`: execution/preservation status, fingerprints and readable report.

Sibling baseline_before/after and superseded style_initial/style_final retain their evidence. Root `verification.json` records final hashes/tests/commits/clean status; `completion.md` mirrors this record locally. Artifacts remain ignored by existing policy, with the readable completion/protocol tracked in Git.

Existing branch `refactor/research-v2`, clean starting HEAD **fb1bc06ce55f7fa6d203d3ab74a7c54f50096952**. Implementation **a2a1d164fe31830801f25cd70563edec48ece590**, `feat(style): add fold-local V2-04 measures and audit`. Diagnostic correction **df839581a389ed67de96cc44e481f5971e1dbe57**, `fix(style): retain serialized RMS discrepancies as audit evidence`. Final documentation commit hash is recorded in verification.json after committing. Scientific provenance records its actual pre-commit checkout/dirty state and links to these verified source hashes, rather than pretending the final docs commit existed at fit time. Git staging/commits used the authorized metadata-write escalation; no automatic approval rejection, branch switch, worktree or push.

Scientific scope/settings did not deviate. Documented adaptations are all-note/fixed-window event profiles, the user-supplied recorded Gaussian floor and acceptable solver implementation choice. The unintended agreement execution gate and initial test-fixture assumptions were corrected without metric tuning; all failed/superseded diagnostic evidence is retained. Native workspace writes were used after the patch tool's spurious reparse-point errors. **Stop after V2-04 for review. Do not start V2-05.**
