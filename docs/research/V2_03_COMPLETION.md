# V2-03 completion record - 2026-10-04

V2-03 content audit only is complete. Stop for review. V2-02, V2-04 and V2-05 were not started. No training, feature extraction, transfer optimization or frozen output regeneration. No push.

## Implementation and exact changed files (10)

- `src/musicians_style/content_metrics.py`: reusable raw MIDI observation and original protected-mask measurements under `v2-03.content.1`; reuses frozen parser, E3 selector and existing polyphony functions.
- `src/musicians_style/content_audit.py`: explicit 600-output / 150-identity audit, root/provenance support, hash preservation, per-output failures and grouped direction summaries.
- `tools/content_audit.py`: thin entry point; ambient import guard before content imports, fresh destination required.
- `tests/unit/test_v2_content_metrics.py`: synthetic identity, pitch/onset/note-off/channel changes, meter/resolution/format, velocity, transposition, same-pitch overlaps, local overlap extent, output Skyline reselection, inserted high notes, selector ties, duplicate occurrences, simultaneous/cross-track order, unmatched notes, empty/SMF2 and invalid-shift cases.
- `tests/unit/test_v2_content_audit.py`: synthetic complete/read-only audits, missing/hash/corrupt/unknown-source/coverage failures, protected destinations, clustered denominators/weights and subprocess import guards.
- `docs/STATUS.md`: current completion/review boundary with preserved V2-01 snapshot.
- `docs/research/ROADMAP.md`: V2-03 completion and explicit audit command; other backlog unchanged.
- `docs/research/RESEARCH_CONTEXT.md`: implemented content contract and diagnosed historical ambiguity.
- `docs/research/EXPERIMENT_REGISTRY.md`: V2-03 completion and artifacts.
- `docs/research/V2_03_COMPLETION.md`: this tracked completion record.

All paths are relative to `D:/Studia/inzynierka_dev`. Scientific artifacts are ignored by Git under the existing repository policy; this completion summary is tracked. Existing pytest configuration, property budgets and all previous tests are unchanged. The two new synthetic test files contribute 42 fast cases; dataset-wide checks remain outside pytest. No new package namespace, adapters, metrics registry or speculative abstractions.

## Content contract and findings

Select the original source's frozen E3 Skyline mask once for each comparison. On/off comparison uses exact tick/channel/pitch and multiplicity, with an allowed recorded shift only for historical E3. Extra accompaniment does not replace the original protected mask. Source selector ties are reported; the original deterministic tie break remains the operational selector.

Protected melodic identity comprises original absolute pitch, onset, note-off, identifiable duration and event order. Same-channel/pitch overlap components make individual durations ambiguous even when observable on/off events remain present; arbitrary FIFO tuple failures cannot prove damage. Observable within-track simultaneous order is checked for V2, while cross-track order and nonunique occurrences are marked ambiguous. Historical E3 retains its original representation's same-tick permutation equivalence and reports literal serialization order as a separate diagnostic. Its original tuple-plus-velocity protection is reproduced separately.

Meter, essential tempo/meter/key/program metadata, raw SMF format/resolution and channel availability are structural/technical checks. Protected attack velocity is a current-transformation invariant and never enters melodic identity status. Velocity is undefined when the protected attack cannot be matched, rather than falsely labelled changed because pitch/timing changed. Note count, end tick and polyphony are validity/structure descriptors, not melody substitutes.

| Assessment | Final result |
|---|---|
| Coverage / execution | 300 E2 + 300 E3 + 150 source-self identities; zero measurement exceptions; all counts/unique tasks/directions verified |
| Historical E3 observable on/off identity | 300 passed; original tuple/velocity protection also 300 passed |
| Historical E3 identifiable-duration assessment | 27 passed / 273 ambiguous; no semantic retention failures |
| Historical E3 literal simultaneous order diagnostic | 7 passed / 37 ambiguous / 256 changed; these are not failures of the historical same-tick permutation-equivalent policy |
| Historical E3 reselected output Skyline | 19 discrepancies: 17 overlap/pairing/reselection cases + 2 higher-note outputs; original protected tuple membership has zero failures |
| V2 absolute-pitch on/off identity on E3 | 67 passed / 233 failed, exactly the zero/nonzero-transposition partition |
| Full V2 melody policy on E3, including strict observable order | 1 passed / 10 ambiguous / 289 failed; 56 of the 67 pitch-retaining outputs also change identifiable simultaneous order |
| V2 on/off identity on E2 | 283 failed / 17 undefined due to unmatched raw note-off events; all 300 remain in artifacts |
| Source-self identity references | Observable events 150 passed; full V2 assessment 4 passed / 146 ambiguous; duration 110 passed / 40 ambiguous. Ambiguity does not mean the unchanged reference was damaged |
| Structural/technical invariants | No failures in either frozen output set or identities within the declared metadata/channel scope |
| Protected attack velocity | Historical E3 300 passed; V2 E3 67 passed / 233 unmatchable; E2 300 unmatchable under the original protected exact attacks |

Work-group bootstrap uses 2,000 draws, seed 1729, percentile 95% intervals, equal weighting of within-group output means. Undefined values are excluded with explicit counts/denominators; errors count as undefined. All six transfer directions per experiment and three composer identity strata are reported, along with pooled descriptive summaries. No aggregate content scalar or new harmonic/phrase tolerance is invented.

Pooled E3 exact on/off complete-retention output rate is 67/300 = 0.223333; the equal-work-group mean is 0.267241 with CI [0.193463, 0.343870] over 87 groups. Historical E3 and source identity group means are 1.0 [1.0, 1.0]. E2 exact complete-retention mean is 0.0 [0.0, 0.0] over 283 defined outputs / 86 groups. Pooled E2 retained on-event fraction mean is 0.002769 [0.001949, 0.003741] and off-event fraction mean 0.000260 [0.000194, 0.000325]. These are distinct event-retention measurements, not a style or perceptual score.

## Exact verification commands and results

Working directory: `D:/Studia/inzynierka_dev`; interpreter prefix for every Python command below: `.venv/Scripts/python.exe`. No dependency installation or environment repair was needed. `-B` avoids bytecode and `-p no:cacheprovider` avoids the existing inaccessible cache.

| Command suffix | Result |
|---|---|
| `-B -m pytest -q -p no:cacheprovider --durations=10` | Final 494 passed / 64 deselected; 14 existing dependency warnings; 24.62 s |
| `-B -m pytest -q -p no:cacheprovider -o addopts= tests/property --hypothesis-profile=dev --durations=5` | 17 passed; 13 existing warnings; 9.95 s; frozen code and properties unchanged |
| `-B -m pytest -q -p no:cacheprovider -o addopts= tests/unit/e3 tests/unit/test_midi_parser.py tests/unit/test_content_metrics.py tests/unit/test_inference_midi_events.py tests/unit/test_v2_content_metrics.py tests/unit/test_v2_content_audit.py tests/unit/test_asset_paths.py tests/unit/test_provenance.py tests/unit/test_research_audit.py tests/unit/test_property_profiles.py --durations=10` | Final 115 passed; 14 existing warnings; 11.47 s; includes explicit existing E3 regression/HTTP/inference fixtures and all 42 new synthetic cases |
| `-B tools/research_audit.py --scope baseline --output experiments/research_v2_03_2026-10-04/baseline_before` | Passed; 3.05 s command wall time; 24 passed / 2 unavailable provenance checks / 0 failed |
| `-B tools/research_audit.py --scope baseline --output experiments/research_v2_03_2026-10-04/baseline_after` | Passed; 4.22 s command wall time; same check results |
| `-B tools/content_audit.py --output experiments/research_v2_03_2026-10-04/content_initial` | Passed; 750 records; recorded audit phase 162.33 s; development artifact, superseded |
| `-B tools/content_audit.py --output experiments/research_v2_03_2026-10-04/content_final` | Passed; 750 records; recorded audit phase 183.19 s; review iteration before historical same-tick interpretation correction, superseded |
| `-B tools/content_audit.py --output experiments/research_v2_03_2026-10-04/content_verified` | Definitive passed artifact; 750 records; recorded audit phase 190.07 s, excluding final artifact serialization |
| `git diff --check` and staged equivalent | Passed |

Development checks first passed 33 synthetic tests and later 41 before the last additional clustered-fraction case. The first test-fixture run had seven missing-parent-directory setup errors; the fixture was repaired and every final check passes. Additional fast/focused reruns followed measurement and historical-interpretation changes; the final results above apply to the committed implementation. Full-budget properties and a new whole-repository expensive regression run were not needed; V2-01 already established broad coverage, and unchanged full-budget retention is included in focused verification.

## Preservation and provenance

The definitive audit hashes 3,052 frozen/input files before and after: complete frozen experiment run trees (including reports/task/config/result artifacts), E1-E4 source plus reused MIDI/evaluation/GA/features source, all configurations, canonical manifest and all source MIDI bytes. No changed, missing or added frozen file was found. Independent V2-01 baseline before/after fingerprints match for 1,004 evidence files, 34 frozen experiment source files, nine configurations and 13 selected PDFs. Post-audit verification rechecks every definitive frozen hash and the audit's new source hashes. AST inspection finds no reverse imports from E1-E4 to new modules.

Actual existing branch: `refactor/research-v2` (the old V2-01 status table's main reference is historical). Base HEAD: `fdf529ff75f7895666959ed40c41ce2b24e7d8cf`. Audit provenance records this pre-commit HEAD and dirty implementation files, exact editable checkout import, interpreter/dependency metadata, roots and configuration/source fingerprints. Implementation commit: `df2f9337c4606d39fbeadbcc24e13524302e0094`, `feat(content): add V2-03 frozen output audit and focused tests`. A separate local documentation commit contains the completion/navigation records. Local artifact `verification.json` records the resulting commit pair; no branch switch or push.

Git staging/commit initially hit the workspace sandbox's read-only Git metadata boundary. The authorized local commit succeeded through the permission escalation; automatic review did not reject it. Documentation updates used normal file writes after the patch tool reported a spurious reparse-point error; native file/parent inspection confirmed ordinary files within the workspace.

## Artifact locations

Definitive scientific artifact directory: `D:/Studia/inzynierka_dev/experiments/research_v2_03_2026-10-04/content_verified`:

- `metric_contract.json`: versioned policy/categories/status meanings and limitations.
- `provenance.json`: import/environment/Git/roots/config/source provenance.
- `per_output.json`: all 750 references/outputs, paths/hashes, missing events, separate policies/invariants and failure/undefined records.
- `ambiguities.json`: explicit subset retaining overlap/order/selector/high-note/undefined diagnostics with task identities.
- `summary.json`: clustered direction/pooled event-retention rates and fractions, full-policy/duration/order/velocity/structural and reselection counts.
- `audit.json`: seven completion/integrity checks, counts and execution errors.
- `frozen_sha256.json`: all 3,052 before/after-verified hashes.
- `report.md`: readable scientific report.

`D:/Studia/inzynierka_dev/experiments/research_v2_03_2026-10-04/baseline_before` and `baseline_after` each contain V2-01's `audit.json`, `provenance.json`, `asset_inventory.json`, `test_inventory.json`, `report.md`. `content_initial` and `content_final` remain inspectable development iterations; use `content_verified` for conclusions. `verification.json` records final preservation/test/commit evidence, and local `completion.md` mirrors this record with final commit hashes.

## Limitations, deviations and stop boundary

No scientific-scope deviation. Historical same-tick equivalence follows frozen `midi/types.py` and printer/parser contracts; literal ordering is a separately assessed V2 constraint, not a retroactive E3 failure. The tracked completion distinguishes observable retention from full identifiable duration/order. Original source-self identity references are read directly; no serialized identity outputs were generated.

MIDI event inclusion cannot establish causal note lineage, uniquely pair overlapping same-pitch voices or resolve identical occurrences/cross-track ties. Skyline is an operational melody approximation, not musicological ground truth. The frozen essential metadata scope covers tempo/meter/key/program fields represented by the parser; other raw meta messages are diagnostic, track allocation/end-of-track are excluded, and other channel messages such as controllers are outside this content contract. The new policy uses ticks and requires unchanged resolution separately. E2's 17 undefined raw-event cases remain explicit rather than being removed or called successes. Cluster intervals are descriptive; no multiplicity adjustment, held-out musical validation, listening claim or hypothesis-confirming p-value is made. Historical environment/provenance gaps remain unrepaired.

No new feature backends, style metrics, objectives, optimization, dataset tests, neural training, frozen source/config/result/report edits or downstream experiment launch. Stop after V2-03 for review.
