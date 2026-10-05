# V2-05 completion record — 2026-10-05

Updated 2026-10-05. V2-05 fixed-budget objective pilot is complete: 18/18 runs completed; execution/preservation passed=True. Stop after V2-05 for review; no full E1c, outer-test evaluation, larger transfer experiment or push. V2-01 through V2-04 remain preserved. See [V2_05_COMPLETION.md](V2_05_COMPLETION.md) and [V2_05_PROTOCOL.md](V2_05_PROTOCOL.md).

## Fixed cohort, provenance and implementation

The user fixed RMS67/Gaussian67/logistic93 using V2-04 evidence, then explicitly confirmed E3 elitism 2, tournament 3 and five mutation sigmas while requiring inactive transposition. This supersedes only the earlier V2-05 provisional two-candidate language and prior stop boundary. No broad replanning.

Frozen repeat 0 / outer fold 0 / inner fold 0; lexical (group_id, sample_id) selection without score/behavior/note-count inspection:

- `bach--fugue-bwv-846` — work `bach--bwv-846`; SHA256 `965a7e3fd012bf555d3732ba36709a0b1f0f013406802c79f71997f6c79c305a`.
- `beethoven--piano-sonatas-13-4` — work `beethoven--piano-sonata-13`; SHA256 `b5be1123ccf8dd9d60709f8efb84b1533e6e37609806381726c5dae194776c86`.
- `chopin--ballades-2` — work `chopin--ballades-2`; SHA256 `ad9d6cb1cd626171b25ccd904f8a8be63793d3fa624013c99fa4239ae57d5016`.

All six directions x three objectives, exactly 18 searches. Fixed seed 1729, population 32, 60 generations, stagnation 61. Initial 32 + 60x30 offspring = 1,832 proposals; identity + 61x32 population requests = 1,953 evaluation requests per run. Actual unique transformed-representation evaluations differ by objective/cache; all 61 histories and inactive-transpose counters are saved.

External adapter executes the original frozen E3 function code with private canonicalization/assertion bindings. No shared-global monkeypatch or frozen E3 source hook/change. All five RNG draws retained; every transformation/evaluation/cache key and selected/history genome has zero transposition. Original Skyline mask protects absolute pitches/onsets/offs/identifiable durations. Reused E3 constraints and V2-03 serialized checks enforce feasibility; same-tick order is diagnostic. Identity retains original bytes; other objectives score saved MIDI, avoiding optimized/saved representation mismatch.

One 76-piece / 45-work inner-training bundle (Bach26, Beethoven31, Chopin19 pieces). Manifest/split hashes exactly match preserved V2-04 fingerprints. All 74 other sample/group/hash identities are forbidden; only training and three selected validation MIDI sources (79 total) were opened for representations/features. No outer-test original MIDI/features were accessed. Historical artifacts were hashed only for preservation, not scored or used for fitting. No full-corpus feature cache or outer-fold fitted model was loaded.

RMS67/Gaussian67/logistic93 and both equal-work event prototypes reuse unchanged V2-04 implementations/settings. No copied metric formula. Historical-configuration RF is freshly fit on the same inner training: 300 trees, balanced, max_features=0.5, min_samples_leaf=2, seed1729, n_jobs=1, VarianceThreshold(0). Reusing frozen outer-training RF would leak validation. Detailed means/std/floors/prototypes/scaler/coefficients/classes/iterations/training/forbidden identities are in fits/inner00.json; RF trees and all fitted objects are in joblib.

Fit persisted at 2026-10-04T23:26:06.418609+00:00; joblib SHA256 665171f9285fe4721b9060daf7a23bb9fd6082d72b6e667e98b95b972a5feec4. Warnings: []. Exact serialized-fit score equivalence checked for all 79 accessed inputs. Both manifest and fits precede optimization_started.json. No output fitting/tuning/calibration/selection.

## Exploratory comparison

| Objective | Exact observable policy | time_pitch positive / negative / null | time_pitch mean | Optimized mean |
|---|---:|---|---:|---:|
| rms67 | 6/6 | 1 / 5 / 0 | -0.022067 | 0.377232 |
| gaussian67 | 6/6 | 2 / 4 / 0 | -0.026364 | 0.026629 |
| logistic93 | 6/6 | 0 / 6 / 0 | -0.068912 | 0.139354 |

Optimized deltas use each metric’s own scale and are not comparable across RMS67/Gaussian67/logistic93. Compare objective variants through the same separately measured time_pitch, consistency and content policy. Exact observable content and structural invariants take priority, followed by primary time_pitch, direction consistency, negative/null rates and disagreements. The optimized score alone is insufficient. Each time_pitch count above is a direction on one fixed source/work, not an independent repeated sample. Full six-direction scores, margin/source-drop movement, null/negative/undefined counts and optimized/non-optimized disagreement are in report.md/per_output.json/summary.json.

All 18 searches improve the optimized objective; time_pitch improves in only 3/18 outputs, with all three objective means negative. onset_duration decreases in all 18 outputs. RMS67 has 1/6 positive primary directions, Gaussian67 2/6, logistic93 0/6. Thus the pilot establishes technical feasibility under the common observable-content policy, but does not establish consistent target-style movement through the structurally distinct diagnostics. Gaussian has more positive directions, RMS a less negative mean; neither descriptive ordering establishes a winner on three fixed works. Stop for review; no objective is promoted.

### rms67

Runtime 5756.51 s; unique evaluations 10101; original-byte identity outputs 0/6; ambiguous duration 6/6.

| Measure | Mean delta | Positive / negative / null / undefined |
|---|---:|---|
| rms67 | 0.3772319450799207 | 6 / 0 / 0 / 0 |
| gaussian67 | 0.017362073266400175 | 6 / 0 / 0 / 0 |
| logistic93 | -0.042927262071495775 | 3 / 3 / 0 / 0 |
| onset_duration | -0.06732617501242211 | 0 / 6 / 0 / 0 |
| time_pitch | -0.02206651273243067 | 1 / 5 / 0 / 0 |
| historical_rf | 0.021307563681961 | 2 / 3 / 1 / 0 |

Optimized-positive but nonpositive time_pitch directions: 5/6. These disagreements remain evidence; no objective is automatically promoted.

### gaussian67

Runtime 5785.76 s; unique evaluations 10328; original-byte identity outputs 0/6; ambiguous duration 6/6.

| Measure | Mean delta | Positive / negative / null / undefined |
|---|---:|---|
| rms67 | 0.33356605425245817 | 5 / 1 / 0 / 0 |
| gaussian67 | 0.026628668638463898 | 6 / 0 / 0 / 0 |
| logistic93 | -0.0005720610772251908 | 4 / 2 / 0 / 0 |
| onset_duration | -0.06320807128913135 | 0 / 6 / 0 / 0 |
| time_pitch | -0.02636408356777979 | 2 / 4 / 0 / 0 |
| historical_rf | 0.027186346908385253 | 4 / 2 / 0 / 0 |

Optimized-positive but nonpositive time_pitch directions: 4/6. These disagreements remain evidence; no objective is automatically promoted.

### logistic93

Runtime 5537.78 s; unique evaluations 10856; original-byte identity outputs 0/6; ambiguous duration 6/6.

| Measure | Mean delta | Positive / negative / null / undefined |
|---|---:|---|
| rms67 | -0.35994618718982857 | 1 / 5 / 0 / 0 |
| gaussian67 | -0.07442117602900093 | 1 / 5 / 0 / 0 |
| logistic93 | 0.13935405326741865 | 6 / 0 / 0 / 0 |
| onset_duration | -0.17757460810498019 | 0 / 6 / 0 / 0 |
| time_pitch | -0.0689123183936763 | 0 / 6 / 0 / 0 |
| historical_rf | 0.034175171642359566 | 5 / 1 / 0 / 0 |

Optimized-positive but nonpositive time_pitch directions: 6/6. These disagreements remain evidence; no objective is automatically promoted.

## Per-run budget, convergence and content diagnostics

| Objective | Direction | Runtime s | Unique evaluations | First final gain generation | Identifiable duration | Strict literal order | Other metadata |
|---|---|---:|---:|---:|---|---|---|
| rms67 | Bach->Beethoven | 315.66 | 1595 | 31 | ambiguous | failed | False |
| gaussian67 | Bach->Beethoven | 295.13 | 1606 | 31 | ambiguous | failed | False |
| logistic93 | Bach->Beethoven | 288.83 | 1824 | 41 | ambiguous | failed | False |
| rms67 | Bach->Chopin | 301.18 | 1618 | 31 | ambiguous | failed | False |
| gaussian67 | Bach->Chopin | 297.11 | 1639 | 11 | ambiguous | failed | False |
| logistic93 | Bach->Chopin | 283.52 | 1806 | 17 | ambiguous | failed | False |
| rms67 | Beethoven->Bach | 1442.27 | 1669 | 3 | ambiguous | failed | False |
| gaussian67 | Beethoven->Bach | 1530.45 | 1829 | 18 | ambiguous | failed | False |
| logistic93 | Beethoven->Bach | 1460.05 | 1832 | 39 | ambiguous | failed | False |
| rms67 | Beethoven->Chopin | 1256.07 | 1726 | 14 | ambiguous | failed | False |
| gaussian67 | Beethoven->Chopin | 1300.22 | 1718 | 33 | ambiguous | failed | False |
| logistic93 | Beethoven->Chopin | 1186.58 | 1812 | 9 | ambiguous | failed | False |
| rms67 | Chopin->Bach | 1547.42 | 1830 | 12 | ambiguous | failed | False |
| gaussian67 | Chopin->Bach | 1453.52 | 1822 | 50 | ambiguous | failed | False |
| logistic93 | Chopin->Bach | 1448.16 | 1830 | 58 | ambiguous | failed | False |
| rms67 | Chopin->Beethoven | 893.91 | 1663 | 55 | ambiguous | failed | False |
| gaussian67 | Chopin->Beethoven | 909.33 | 1714 | 5 | ambiguous | failed | False |
| logistic93 | Chopin->Beethoven | 870.64 | 1752 | 38 | ambiguous | failed | False |

Each run used exactly 1,832 proposals and 1,953 evaluation requests; convergence columns report first attainment of the final saved gain, not an earlier stopping point. Essential metadata and structural requirements are hard checks. Other nonessential metadata may change during existing serialization and are disclosed above; literal same-tick order is diagnostic under the user-fixed policy.

## Inactive transposition and complete per-output style movement

All 18 selected genomes and all 1,098 history genomes have transpose_semitones=0. The adapter discarded 20934 nonzero historical transpose values before transformation/evaluation. This retains the inherited first mutation sigma and all RNG draws without allowing global transposition. Per-run counters are in result.json.

| Objective | Direction | RMS67 | Gaussian67 | Logistic93 | time_pitch | onset_duration | RF |
|---|---|---:|---:|---:|---:|---:|---:|
| rms67 | Bach->Beethoven | 0.037172878521014585 | 0.013861760804699474 | 0.0030738770373270607 | -0.007408267533781099 | -0.026104305710827225 | 0.02236384133023321 |
| gaussian67 | Bach->Beethoven | 0.037172878521014585 | 0.013861760804699474 | 0.0030738770373270607 | -0.007408267533781099 | -0.026104305710827225 | 0.02236384133023321 |
| logistic93 | Bach->Beethoven | -1.2216228590884466 | -0.14253218960353453 | 0.3451095032227678 | -0.07168347233286188 | -0.3491234147413731 | 0.06591715564123883 |
| rms67 | Bach->Chopin | 0.029013186225138377 | 0.009650392302083532 | -1.7126183843452155e-05 | -0.001190786551086398 | -0.016465668338157657 | -0.0033333333333333335 |
| gaussian67 | Bach->Chopin | 0.00841709027262394 | 0.0018769383332779555 | -3.880738978987916e-06 | 0.0022621526776153145 | -0.0010917134568246833 | -0.0033333333333333335 |
| logistic93 | Bach->Chopin | -0.8081511710590309 | -0.06691151431940079 | 8.861157584433697e-05 | -0.03364343362917399 | -0.11324181962803237 | 0.007900876722132763 |
| rms67 | Beethoven->Bach | 0.02112956775103425 | 0.02012233080309067 | 0.0020038787684628426 | -0.0001429050497276929 | -0.0042418996066555525 | 0.0 |
| gaussian67 | Beethoven->Bach | 0.004421146205569992 | 0.052703605340837734 | 0.004620190602168658 | -0.05856138934676969 | -0.1847907348768808 | -0.0006529209621993127 |
| logistic93 | Beethoven->Bach | -0.35837809867245096 | -0.028205580296222 | 0.06864698149636492 | -0.0736863553259971 | -0.07427450389719437 | 0.026700677243797093 |
| rms67 | Beethoven->Chopin | 0.02917399686658717 | 0.010332503625506861 | 0.00040872147951550675 | 0.020496085749946436 | -0.007738943685119182 | 0.12946238078883063 |
| gaussian67 | Beethoven->Chopin | 0.028464352888119104 | 0.010457907221443241 | 0.0004359960483806311 | 0.021002431272085342 | -0.008428987991695336 | 0.12946238078883063 |
| logistic93 | Beethoven->Chopin | -0.3044757426063074 | -0.09829352862571983 | 0.03640572736220198 | -0.05232238880901363 | -0.14771609902008231 | 0.11998442540193743 |
| rms67 | Chopin->Bach | 2.0461098992464786 | 0.03314128620412582 | -7.632536741987656e-05 | -0.12799316219868262 | -0.28211872843681984 | -0.004127744022390328 |
| gaussian67 | Chopin->Bach | 2.019774781346282 | 0.06630897727628804 | 0.0005249422353320397 | -0.11518880683051014 | -0.15554840082046395 | 0.013211446960113645 |
| logistic93 | Chopin->Bach | 1.948480885753758 | 0.042521004890241465 | 0.003168536483012789 | -0.09997825454800502 | -0.2034742924724433 | 0.016544780293446977 |
| rms67 | Chopin->Beethoven | 0.1007921418692711 | 0.017064165858894698 | -0.26295659816301675 | -0.016160040811252663 | -0.06728750429695318 | -0.01651976267157418 |
| gaussian67 | Chopin->Beethoven | -0.09685392371886026 | 0.014562822854236934 | -0.012083491647580547 | -0.00029062164531845625 | -0.0032842848780960665 | 0.002066666666666661 |
| logistic93 | Chopin->Beethoven | -1.4155301374664933 | -0.15310524821936988 | 0.38270495946432004 | -0.08216000571700621 | -0.17761751887075572 | -0.031996885448395684 |

Input/output affinities, source drop, target-source margin and all content measurements are saved separately per output; these target-affinity deltas are not a combined style/content score.

## Rejected-candidate diagnostics

| Objective | Direction | Rejection reason counts |
|---|---|---|
| rms67 | Bach->Beethoven | `{"note_count": 244, "polyphony": 33}` |
| gaussian67 | Bach->Beethoven | `{"note_count": 262, "polyphony": 32}` |
| logistic93 | Bach->Beethoven | `{"note_count": 532, "polyphony": 233}` |
| rms67 | Bach->Chopin | `{"note_count": 238, "polyphony": 34}` |
| gaussian67 | Bach->Chopin | `{"note_count": 296, "polyphony": 35}` |
| logistic93 | Bach->Chopin | `{"note_count": 635, "polyphony": 131}` |
| rms67 | Beethoven->Bach | `{"note_count": 285, "polyphony": 335}` |
| gaussian67 | Beethoven->Bach | `{"note_count": 480, "polyphony": 431}` |
| logistic93 | Beethoven->Bach | `{"note_count": 524, "polyphony": 318}` |
| rms67 | Beethoven->Chopin | `{"note_count": 429, "polyphony": 702}` |
| gaussian67 | Beethoven->Chopin | `{"note_count": 401, "polyphony": 697}` |
| logistic93 | Beethoven->Chopin | `{"note_count": 521, "polyphony": 804}` |
| rms67 | Chopin->Bach | `{"note_count": 421, "polyphony": 347}` |
| gaussian67 | Chopin->Bach | `{"note_count": 515, "polyphony": 485}` |
| logistic93 | Chopin->Bach | `{"note_count": 526, "polyphony": 350}` |
| rms67 | Chopin->Beethoven | `{"note_count": 340, "polyphony": 1128}` |
| gaussian67 | Chopin->Beethoven | `{"note_count": 437, "polyphony": 1114}` |
| logistic93 | Chopin->Beethoven | `{"note_count": 498, "polyphony": 1132}` |

These counts describe failed constraint/content checks among evaluated proposals. Multiple reasons can refer to the same candidate, so their sum is not a distinct rejected-candidate total. All unique evaluations, including infeasible ones, remain in the reported search accounting. Selected-output feasibility is checked separately.

## Verification and preservation

Working directory D:/Studia/inzynierka_dev; original .venv/Scripts/python.exe, Python 3.10.20; no installs/upgrades. -B prevents bytecode; pytest cacheprovider disabled.

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --durations=10` | 558 passed / 70 deselected, 14 existing warnings; 37.61 s (concurrent load) |
| `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider -o addopts= tests/unit/test_objective_pilot.py tests/unit/test_style_metrics.py tests/unit/test_style_audit.py tests/unit/e3 tests/unit/test_v2_content_metrics.py tests/unit/test_midi_parser.py --durations=10` | 118 passed, 14 existing warnings; 20.35 s |
| `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider -o addopts= tests/unit/test_objective_pilot.py` | 10 passed, 13 existing warnings; 5.82 s, includes final synthetic full-runner fixture |
| `.venv/Scripts/python.exe -B tools/objective_pilot.py --output experiments/research_v2_05_pilot_2026-10-05` | passed=True; 18/18 completed; 17156.71 s total measured pilot wall time |

Synthetic repeated same-seed searches test all three objectives, large active historical sigma with zero transformation/evaluation/history genomes, frozen-global preservation, objective equivalence, content order/duration distinction, fixed budget, score/convenience-independent selection, fit-before-search sequencing, no test-source reads and fresh-destination rejection. Nine fast cases plus one integration-marked synthetic runner case. Fast/focused suites ran before the final integration fixture was added; the subsequent 10-case suite includes it. No test tiers or existing tests changed.

Before/after byte fingerprints: 3703 protected files, changed=[], added=[]. Includes frozen E1-E4 source/config/results/reports, prior V2 scientific artifacts/code/completion/protocols, reused implementations/tests, and the 79 accessed original sources. Outer-test original files were not opened to hash them; no operation writes to dataset paths. AST reverse-import scan is empty. Final artifact/source hash verification and git diff checks are in verification.json. All E1-E4 remain frozen, including transposition-allowed historical E3. Prior full baseline audit need not be rerun across outer-test originals for this pilot.

## Artifacts, exact changes and local commits

Definitive scientific directory: D:/Studia/inzynierka_dev/experiments/research_v2_05_pilot_2026-10-05. Contains protocol/provenance/cohort/source/input hashes, fit provenance/model hashes, optimization-start sequence, 18 output MIDIs/results/histories, all separate style/content scores, summary, readable report and preservation audit. Command/test logs and verification/completion copy: experiments/research_v2_05_2026-10-05. Scientific artifacts remain ignored under existing Git policy.

- `src/musicians_style/objective_pilot.py`
- `tools/objective_pilot.py`
- `tests/unit/test_objective_pilot.py`
- `docs/research/V2_05_PROTOCOL.md`
- `docs/research/V2_05_COMPLETION.md`
- `docs/STATUS.md`
- `docs/research/ROADMAP.md`
- `docs/research/RESEARCH_CONTEXT.md`
- `docs/research/EXPERIMENT_REGISTRY.md`

Existing branch refactor/research-v2; starting HEAD 5d9cbab. Implementation commit cc1051a; documentation commit recorded after creation in verification.json. Scientific provenance records actual pre-commit HEAD/dirty state and precise source hashes, not a fabricated post-completion commit. No branch/worktree change or push.

## Limitations and deviations

No scientific scope/budget deviation. Numerical inheritance was clarified with the user before implementation. Local Git metadata writes used authorized escalation. No source hook, new operator/framework/registry, E1c or outer-test evaluation. Exactly 18 real optimizations, no discarded scored development cohort or output-based tuning.

Overlapping same-pitch voices can leave durations ambiguous despite preserved observable on/off events and frozen protected tuples; these are never called certified identifiable durations. Strict historical V2-03 order/status fields remain diagnostic under the review clarification. Essential metadata/channel/format/resolution and velocity are separately checked. Skyline/FIFO/essential-metadata scope and event-profile piano/meter/bin adaptations remain V2-03/V2-04 limitations.

Logistic is non-independent when optimized; RF/logistic share custom93/corpus. RMS/Gaussian share 67 components, but both link and variance differ. Event profiles are structurally distinct, not perceptual ground truth. Lexical source/form selection is fixed and may be unrepresentative; this historically inspected validation corpus is not a fresh untouched confirmatory test. No p-values/confirmatory bootstrap or listening claim. Only synthetic optimizations were repeated for determinism; real 18-run same-seed reruns remain unmeasured. Serialized real models/scores and output bytes were verified exactly. Runtime reflects actual sequential execution and concurrent initial tests.

Stop after V2-05 for review. Do not automatically launch full E1c, a 300-task transfer experiment, neural/sequence/StyleRank work or outer-test evaluation.
