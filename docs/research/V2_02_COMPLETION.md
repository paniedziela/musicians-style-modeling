# V2-02 completion record — 2026-10-04

V2-02 feature feasibility only is complete. Stop for review. Full E1c classification, V2-04 and V2-05 were not started; no broad replanning, other backend, training, transfer optimization, MusicXML substitution or push.

## Pre-implementation V2-03 clarification

Recorded in RESEARCH_CONTEXT before any feature implementation: ordering of distinct simultaneous same-tick MIDI events remains available as a diagnostic, but does not by itself constitute a hard melodic-identity violation. Exact protected absolute pitch, onset and note-off / identifiable duration remain the primary hard requirements. This clarification is reflected in current navigation/policy documents only. Completed V2-03 code, completion record and scientific artifacts retain their original strict-order contract and statistics and were not rewritten or regenerated.

## Exact changed files (13)

All paths below are relative to `D:/Studia/inzynierka_dev`.

- `src/musicians_style/feature_backends/__init__.py`: opt-in adapter package only.
- `src/musicians_style/feature_backends/custom93.py`: imports frozen E1b extraction, schema version and FEATURE_SPECS; no copied/moved feature implementation. Names, order, units, families, float64 semantics and exported matrix order remain exact.
- `src/musicians_style/feature_backends/musif.py`: pinned musif 1.2.4 MIDI adapter, serial extraction with caching/repeat expansion disabled, reviewed score-level musical numeric projection and explicit nullable/missing/nonfinite diagnostics. No category or metadata encodings.
- `src/musicians_style/feature_feasibility.py`: frozen training-only selection, manifest before extraction, descriptive MIDI sanitation, per-sample subprocess/timeouts/failures, two repetitions, full schema/value/status comparisons, separate numeric matrices with sidecar row mapping, runtime/coverage/family inventories, explicit dataset-wide custom93 equivalence audit and preservation hashes. Reuses existing root/provenance and frozen-file inventory helpers.
- `tools/feature_feasibility.py`: thin entry point with ambient checkout import guard and fresh/protected destination checks; no automatic import repair.
- `tools/feature_backend_worker.py`: one MIDI extraction per isolated subprocess, worker checkout guard, versions/environment and explicit exception record. Backend logs are kept alongside the record.
- `requirements/research-v2-02-musif.lock.txt`: all 89 resolved environment distributions, including installer/build tools; Windows/Python 3.10 lock, separate from frozen project requirements.
- `tests/unit/test_feature_feasibility.py`: 26 synthetic fast cases for train/group selection, leakage rejection including numeric identifiers, metadata/tick preservation, exact wrapper/cache contract and order, missing/nonfinite/invalid results, nullable union mapping, repeated outcomes/failures, worker failure, stale/foreign/missing import guards, all-attempt coverage under missing/hash-changed input, read-only behavior, protected/existing destinations and environment isolation. No real dataset-wide pytest test.
- `docs/STATUS.md`: current V2-02 completion and review boundary, with V2-01 historical snapshot retained.
- `docs/research/ROADMAP.md`: V2-02 completion/command and same-tick clarification; remaining task scope/candidates/budgets unchanged.
- `docs/research/RESEARCH_CONTEXT.md`: clarification recorded first, then actual feature findings, isolated environment and limitations.
- `docs/research/EXPERIMENT_REGISTRY.md`: separate completed feasibility identity and artifacts; full E1c remains proposed.
- `docs/research/V2_02_COMPLETION.md`: this exact completion record.

Frozen E1-E4 source, configurations, schemas, results/reports/checkpoints and original requirements/pyproject remain unchanged. No reverse import from frozen experiments to the adapters. No branch switch or worktree creation.

## Predeclared pilot and measured findings

Selection: repeat 0 / outer fold 0 **training data only**; first three lexical work groups per composer, then one lexical sample per work. Identities, source paths/hashes and frozen manifest/split/cache hashes were persisted in `pilot_manifest.json` before extraction; there was no success-based selection or sample exclusion. The outer test partition was never used for the musif pilot. The separately authorized custom93 equivalence audit covers the complete accepted 150-sample dataset, including its test rows, solely to verify the frozen cache; no model is fitted.

| Predeclared sample | Work group | Raw musif columns | Retained numeric columns |
|---|---|---:|---:|
| bach--fugue-bwv-846 | bach--bwv-846 | 1345 | 203 |
| bach--fugue-bwv-848 | bach--bwv-848 | 1605 | 255 |
| bach--fugue-bwv-854 | bach--bwv-854 | 1435 | 221 |
| beethoven--piano-sonatas-1-1 | beethoven--piano-sonata-1 | 1693 | 283 |
| beethoven--piano-sonatas-10-1 | beethoven--piano-sonata-10 | 1615 | 265 |
| beethoven--piano-sonatas-12-1 | beethoven--piano-sonata-12 | 1707 | 301 |
| chopin--ballades-1 | chopin--ballades-1 | 1847 | 321 |
| chopin--ballades-2 | chopin--ballades-2 | 1683 | 285 |
| chopin--ballades-4 | chopin--ballades-4 | 1863 | 329 |

| Final assessment | custom93 | musif |
|---|---|---|
| Coverage | 9/9 samples, 18/18 successful attempts | 9/9 samples, 18/18 successful nullable results |
| Explicit failure records | 0 in definitive pilot | 0 in definitive pilot; failures.json is an explicit empty list |
| Repetition agreement | Exact schemas/values/statuses/quality diagnostics for all 9 pairs | Exact schemas/values/statuses/quality diagnostics for all 9 pairs |
| Feature counts per sample | 93, original fixed order | 203, 221, 255, 265, 283, 285, 301, 321, 329 |
| Pilot union / intersection | 93 / 93 | 405 / 187 |
| Finite common features / constants among them | 93 / 0 | 186 / 23 |
| Missing/nonfinite values | None | NumericTempo missing in all 18 attempts; no other retained missing values or infinities |
| Complete numeric cases without missing values | 18/18 | 0/18; this is a valid nullable extraction result, not a ready classifier matrix |
| Total recorded extraction seconds | 3.155 | 217.869 |
| Total wall seconds (musif includes subprocess startup/sanitation) | 3.155 | 231.220 |

custom93 family counts: pitch 16, melody 31, rhythm 16, texture 7, harmony 17, structure 6. musif retained union: melody 346, scale 42, core 5, rhythm 4, dynamics 4, tempo 2, density 2. Raw musif rows contain 1,345-1,863 columns; the adapter excludes all identifiers/names/paths, musical category strings and part/sound/family headers. All excluded column names are diagnostics, never matrix entries. Ambitus, texture and key were requested but their part/category outputs are not retained by this bounded score-level numeric adapter; this is not the full musif feature set.

`NumericTempo` is missing despite MIDI containing tempo messages: this is an upstream MIDI-extraction limitation, not silently repaired with custom93 or XML data. The four retained stock dynamics descriptors are constant over these nine MIDI inputs; no expressive-velocity claim follows. The other constant/common-column names are listed in summary.json. No constants or missing columns are removed, zero-filled or fitted away. Observed interval vocabulary changes row schemas; feature_cache.json records the descriptive union, explicit absent-column/null entries and row statuses. That union is **not** a learned full-E1c vocabulary. custom93's exported matrix separately preserves its frozen FEATURE_SPECS order.

The explicit audit `custom93_equivalence_audit.json` passes for **150/150** accepted rows: exact schema version e1.3.0, names/order/group/unit, identity coverage, source hashes and exact numeric values, with no tolerance. Recorded dataset-wide audit runtime: 17.986 seconds. Each mismatch/error would remain a per-sample audit record; this computation is not an ordinary pytest test.

## Dependency/environment changes

Original interpreter remains `.venv/Scripts/python.exe`, Python 3.10.20. Before/after dependency metadata is identical; musif/music21 remain absent there. No upgrade or edit to existing dependencies, requirements.txt or pyproject.toml. A new independent virtual environment was created at `D:/Studia/inzynierka_dev/experiments/research_v2_02_2026-10-04/musif_env` without system site packages. It contains musif 1.2.4, music21 9.9.2, numpy 2.2.6, scipy 1.15.3, pandas 2.3.3, ms3 2.4.2, webcolors 1.12 and their transitive dependencies. The tracked lock contains all 89 installed distributions and is fingerprinted in pilot provenance. Worker records retain the actual executable, Python, prefixes, checkout import and complete installed version map.

No torch or editable project dependency installation was added to the musif environment. The orchestrator intentionally sets checkout-first PYTHONPATH only for its isolated worker and records the worker's verified resolved package path. Every worker uses a fresh neutral `input.mid` in its own attempt directory, with descriptive text/meta removed and retained message ticks preserved. No music21/musif global configuration file is edited. Logs/caches are local attempt artifacts or normal temporary files; no original MIDI is overwritten.

Exact setup commands from the checkout:

```powershell
.venv/Scripts/python.exe -m venv experiments/research_v2_02_2026-10-04/musif_env
experiments/research_v2_02_2026-10-04/musif_env/Scripts/python.exe -m pip install musif==1.2.4
experiments/research_v2_02_2026-10-04/musif_env/Scripts/python.exe -m pip check
experiments/research_v2_02_2026-10-04/musif_env/Scripts/python.exe -m pip freeze
experiments/research_v2_02_2026-10-04/musif_env/Scripts/python.exe -m pip freeze --all
```

pip check: **No broken requirements found**. Initial installation under the restricted network failed with WinError 10013; the authorized isolated installation succeeded through escalation. No automatic approval rejection occurred. Git metadata writes likewise used the authorized escalation. No additional user approval or installation into the frozen environment was needed. A new environment can be reproduced by creating the same Python 3.10 venv and installing `-r requirements/research-v2-02-musif.lock.txt` inside it. The lock is platform-specific; portability to other Python/platform versions was not tested.

Official primary sources consulted for adapter/configuration expectations: [musif tutorial](https://musif.didone.eu/Tutorial.html), [configuration example](https://musif.didone.eu/Config_extraction_example.html), [upstream repository/changelog](https://github.com/DIDONEproject/musif). The website's displayed documentation version is older than the installed wheel; actual 1.2.4 extractor/configuration/constants/handlers were inspected locally, including categorical MeanInterval and KeySignature semantics. L0034/L0230 matrix entries were read for scope; no classifier or StyleRank implementation was started.

## Exact verification commands and results

Working directory: `D:/Studia/inzynierka_dev`. All primary Python commands use `.venv/Scripts/python.exe`; `-B` prevents bytecode writes and `-p no:cacheprovider` avoids the existing inaccessible pytest cache.

| Command suffix | Result |
|---|---|
| `-B -m pytest -q -p no:cacheprovider --durations=10` | **520 passed / 64 deselected**, 14 existing warnings; **32.68 s**. Ran concurrently with the focused suite and pilot; no timing assertion or test-tier changes. |
| `-B -m pytest -q -p no:cacheprovider tests/unit/test_feature_feasibility.py tests/unit/e1/test_composition_features.py tests/unit/test_midi_parser.py tests/unit/test_asset_paths.py tests/unit/test_provenance.py --durations=5` | **63 passed**, 13 existing warnings; **10.03 s**. |
| `-B tools/research_audit.py --scope baseline --output experiments/research_v2_02_2026-10-04/baseline_before` | Passed; 24 passed / 2 unavailable historical-provenance checks / 0 failed. |
| `-B tools/research_audit.py --scope baseline --output experiments/research_v2_02_2026-10-04/baseline_after` | Same passed check results; all baseline fingerprints match. |
| `-B tools/feature_feasibility.py --output experiments/research_v2_02_2026-10-04/pilot_final --musif-python experiments/research_v2_02_2026-10-04/musif_env/Scripts/python.exe` | **Passed**; all 36 attempts; all 18 paired comparisons; exact 150-row cache audit; no protected-file changes. Default timeout 180 seconds per musif worker; no timeout occurred. |
| `git diff --check`, staged equivalent and AST parsing of six new source files | Passed. |

Earlier focused development selections passed 23, 37 and 40 cases; the final commands above apply to committed code. No full-budget properties, full expensive regression suite or E1c classification was needed for these adapters; original tests/tier configuration/property budgets are unchanged. The observed default duration is above the historical below-30-second development target under concurrent load; no broad test replanning was performed.

The same exact feasibility command also ran with `--output experiments/research_v2_02_2026-10-04/pilot_initial` and then `--output experiments/research_v2_02_2026-10-04/pilot_verified`. Each development run completed the same nine samples, both repetitions and explicit custom93 audit. Initial musif records all failed the adapter's numeric validation because KeySignature/MeanInterval were mistakenly admitted as numbers; those categorical fields were excluded after inspecting upstream source. The second run succeeded, followed by a review correction to preserve custom93 order in the aligned exported matrix, schema-version auditing and missing-source coverage. It is superseded by pilot_final. Early implementation hashes are historical development provenance and do not describe final source. All development attempts/artifacts remain inspectable. In total, three completed development/final runs used 54 attempts per backend and three 150-row equivalence audits; acceptance/findings above use only the definitive final run.

## Preservation, provenance and local commits

Final pilot inventories/rechecks **3,099** protected frozen/input files: frozen E1-E4 run trees including scientific artifacts, reused MIDI/evaluation/GA/features source, configurations, derived JSONs, 150 original source MIDIs, completed V2-03 artifact tree and its metric/audit/entry code and completion record. Zero changed/missing/added protected files; unavailable input list is empty. Post-run checks reverified all 3,099 hashes and all seven new source/lock hashes in final provenance. Before/after V2-01 baseline fingerprints match for 1,004 evidence files, 34 frozen experiment source files, nine configurations and 13 selected papers. Original environment versions/import remain unchanged. Historical provenance gaps remain explicit and unrepaired.

Existing branch: `refactor/research-v2`; starting HEAD `f12002b`. Implementation plus the recorded pre-implementation clarification: **`88e837fc64a3905e580a00d981cb2c1bceb7e0ff`**, `feat(features): add V2-02 MIDI feasibility and custom93 cache audit`. The subsequent local documentation commit completes navigation/findings and this record; its exact hash and final clean status are stored in `experiments/research_v2_02_2026-10-04/verification.json`. Final audit provenance records the pre-commit implementation checkout/dirty state; verification.json links it to the resulting commits. No push.

## Artifact locations and limitations

Definitive directory: `D:/Studia/inzynierka_dev/experiments/research_v2_02_2026-10-04/pilot_final`:

- `pilot_manifest.json`: fixed nine identities/work groups/MIDI hashes and train selection, before extraction.
- `backend_contract.json`, `provenance.json`: adapter scope/configuration, roots, source/lock/input hashes and actual primary environment.
- `attempts.json`: all 36 success/failure records, native schemas/values/quality diagnostics, times and musif worker environment.
- `schemas.json`: per-attempt complete schemas and fingerprints.
- `feature_cache.json`: separate numerical matrices and explicit nullable/absent/status diagnostics; identities/labels stay in the sidecar manifest.
- `failures.json`: explicit zero-failure definitive inventory.
- `determinism.json`: all 18 pairs and complete-pilot-schema equality for both repetitions/backends.
- `summary.json`: counts, family coverage, union/intersection/constants, missingness and runtimes.
- `custom93_equivalence_audit.json`: all 150 exact comparisons with explicit per-sample outcomes.
- `frozen_sha256.json`, `audit.json`, `report.md`: preservation/completion checks and readable findings.
- `musif_r0_00` through `musif_r1_08`: individual neutral MIDI input, worker/result/attempt JSON and logs; stock musif logs remain here.

Sibling `baseline_before` / `baseline_after` contain the explicit read-only baseline provenance/audit reports. `pilot_initial` / `pilot_verified` are retained superseded development iterations. `environment_freeze.txt` records the isolated initial freeze. `verification.json` records final tests, fingerprints/commit linkage; `completion.md` mirrors this tracked record locally. Scientific outputs and musif_env remain ignored under existing repository policy; they are not embedded in Git commits.

No scientific-scope deviation. Deliberate adaptation: MIDI-only, score-level nullable numerical musif subset, excluding musical category and instrument-dependent headers as well as descriptive metadata/IDs; this is not upstream's full default feature matrix. Missing tempo, music21 MIDI interpretation not shown equivalent to the frozen parser, per-sample vocabulary, 23 constant common pilot features and Windows-specific dependency resolution limit downstream use. Determinism here establishes repeatability for these nine inputs/environment, not portability or discriminative/perceptual validity. No MusicXML fallback, learned missingness correction, fixed full-dataset vocabulary, model selection or classifier comparison is inferred. No E1c, V2-04 or V2-05 authorization follows from successful extraction. **Stop after V2-02 for review.**
