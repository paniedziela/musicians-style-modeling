# A+B implementation and verification report - 2026-10-05

## Review and commit checkpoint - 2026-10-05

The user has now authorized committing/pushing the implementation and running
Track A only under [the reviewed A protocol](AB_A_PROTOCOL.md). The original
implementation-only report below remains the historical verification record.
Logical code commits: Track A `d3c9fa7`; Track B `a3f6ac3`. Track A was staged as
a standalone backend/evaluator/workflow before B added its operator and modes.
The final shared workflow matches the reviewed A+B source; no scientific feature,
preprocessing or classifier settings changed. Added `tests/unit/test_research_a.py`
extracts independent A workflow checks for the A commit. Git attributes preserve
both original pinned files byte-for-byte; the local commit was corrected before
any push when Git newline normalization was detected. One extra blank EOF line
in the new property test was removed before push; no existing test changed.

Current verification: default 602 passed / 73 deselected (32.39 s), standalone A
20 passed, B/workflow/full property 30 passed. No real A/B execution occurred
during these checks. A preflight confirms train76/45 works, validation44/24 works,
30 forbidden outer-test originals, and unchanged 42/638 backend pins.
A execution follows a clean reviewed commit push; B remains explicitly blocked.


Implemented on `refactor/research-v2`; changes remain local and uncommitted for review.
**No real A comparison, B optimization/pilot, outer-test source/feature access,
main modification, merge or push.** Only the declared nine-source jSymbolic
feasibility set and synthetic software verification ran. Tests are evidence of
software correctness, not musical validity.

## Changes

New modules:

- `src/musicians_style/feature_backends/jsymbolic.py`: ACE XML numeric projection;
  fixed feature names/dimensions, nullable values, explicit malformed/schema errors.
- `src/musicians_style/jsymbolic_runtime.py`: Java/path/process isolation, pinned
  distribution/config/schema verification, neutral MIDI copies, timeout/logs and
  persistent failure records, including zero-exit extraction failures.
- `src/musicians_style/external_classifier.py`: train-only nullable-feature median
  imputation, VarianceThreshold, scaling and the existing V2-04 logistic settings;
  failed/matched denominators and sample/work/hash leakage guards.
- `src/musicians_style/accompaniment_search.py`: immutable raw source bytes and
  original Skyline mask; stable original NoteId/pitch map; single-note +/-4 moves;
  exact raw-message preservation, collision/overtopping/round-trip checks;
  seeded bounded visited-state strict-improvement search and identity fallback;
  read-only frozen logistic93 DEVELOPMENT objective.
- `src/musicians_style/research_ab.py` plus `tools/research_ab.py`: explicit
  feasibility/synthetic workflows and review-gated later A/B commands. A persists
  matched train-only external/custom fits before validation/output scoring; B
  freezes both the external evaluator and V2-05 objective before 48 searches.
  Source fingerprints, failure records, histories, content, accounting, matched
  work summaries and paired/direction comparisons accompany later runs.

New configuration/lock: `configs/research_ab_jsymbolic.txt`,
`configs/research_ab_jsymbolic_schema.json`,
`configs/research_ab_review.template.json`,
`requirements/research-ab-jsymbolic.lock.json`.
New tests: `tests/unit/test_accompaniment_search.py`, `test_jsymbolic.py`,
`test_research_ab.py`, `tests/property/accompaniment/test_pitch_invariants.py`.
New documentation: [contract/reuse ledger](AB_IMPLEMENTATION_CONTRACT.md) and this
report. Five current navigation documents have a new checkpoint paragraph;
historical protocols, completions, decision, results, existing tests and E1-E4
sources/configuration are unchanged.

## Reuse and deliberate deviations

The per-function assumptions and independent tests are in the contract. E3's
`analyse_structure`/`NoteId` are used only on the original source, for deterministic
operational Skyline protection and duplicate-aware identities. IDs contain the
ORIGINAL pitch and are never rebuilt on candidates. `MidiParser` supplies FIFO
parsed tuples; raw on/off locations separately preserve event lineage.
`validate_constraints` is an additional zero-shift check with `roundtrip=False`;
the new raw serializer and exact field checks are stronger than its historical
length/count tolerance. V2-03 raw content checks separately retain identifiable/
ambiguous durations and diagnostic simultaneous order.

Frozen E3's printer is deliberately **not used to serialize B edits**: the new
adapter copies original MIDI messages and changes only the paired on/off note
numbers. Track placement, delta times, metadata, controllers, on/off velocity,
channels and message counts remain intact. Identity returns the original bytes.
Frozen E3 proposal/mutation/genotype and metrics are absent from new pitch search.
The named future E3 comparator uses the already-tested V2-05 zero-transpose
adapter exclusively for the comparator. Synthetic differential tests retain
historical global transposition behavior and identity. No historical scientific
outputs are regenerated or rescored for this correctness check.

Intentional literature deviations:

1. **L0103 inspired/adapted, not reproduced**: polyphonic piano accompaniment
   rather than monophonic bass with fixed chord labels; original protected melody;
   only pitch replacement, no duration adjustment, split/join or global transpose.
2. Pitch radius is anchored to the ORIGINAL pitch, preventing iterative drift;
   final cumulative map size <= `floor(0.20 * accompaniment_count)` by default.
   Restoring an original pitch removes that entry. Small sources with fewer than
   five accompaniment notes consequently have no default nonidentity move.
3. Added MIDI-range, new same-channel/pitch overlap (including cross-track),
   protected attack overtopping, exact serialization/tuple-lineage constraints.
   Existing overlap pairs may remain; unmatched raw events disable nonidentity
   edits but do not invalidate byte-preserving identity.
4. Full visited memory and seeded strict scalar improvement within a fixed budget;
   no complete tabu/Pareto search, local-optimum escape, chord grammar, Markov or
   counterexample objectives. logistic93 is the existing DEVELOPMENT objective;
   jSymbolic never enters B optimization. Numerical improvement tolerance 1e-12.
5. L0034/jSymbolic: stable **jSymbolic 2.2 user distribution**, modern 42-feature
   subset/638 dimensions rather than the paper's historical 111/12 features;
   grouped inner split and existing sklearn logistic configuration rather than
   WEKA LogitBoost/SVM, FuX VNS or contrapuntal costs. No reproduction claim.
6. Whole-piece explicit XML/CSV configuration, vector components flattened in
   pinned upstream order. Dataset IDs/labels, direct instrumentation and expressive
   fields excluded. Remaining constant columns are variance-filtered in TRAIN.
   Nullable external features use TRAIN median imputation; entirely missing train
   columns become zero and are variance-excluded. Real feasibility fits nothing.
7. JVM English/US locale is explicit: a synthetic default Polish-locale attempt
   exposed comma-decimal XML and zero-exit CSV failure. Locale isolation fixes
   export without changing system settings. Java heap ceiling 2 GiB; Java/sklearn
   CPU backends retained. GPU availability/settings are untouched; no compatible
   GPU backend for these frozen algorithms was substituted or benchmarked.

Pinned archive SHA256:
`e1686a6bb1133cc21fb4678185d81e5e0a322b371ac713cbaf5fd9b69e2396cb`.
JAR/library/config/schema hashes are in the lock. Actual executable/version/SHA
are recorded for every attempt: Java 1.8.0_291, build 1.8.0_291-b10.
The official [release distribution](https://sourceforge.net/projects/jmir/files/jSymbolic/jSymbolic%202.2/),
[CLI](https://jmir.sourceforge.net/manuals/jSymbolic_manual/commandline_files/commandline.html)
and [configuration specification](https://jmir.sourceforge.net/manuals/jSymbolic_manual/configuration_files/configuration.html)
were checked; local L0034/L0103 and relevant L0074 sections informed the contract.

## Verification

- jSymbolic: **18/18 successes**, nine predeclared r00/f00 outer-training sources
  twice, **638 fields**, zero missing/nonfinite values, exact schema/value/
  diagnostic repeatability; 52.20 s measured extraction time. No classifier fit,
  full-corpus cache, comparison or search. `experiments/research_ab_feasibility_2026-10-05`.
- Synthetic pitch examples: three repeated 64-request cases (chords, SMF0/channel7,
  zero duration), with before/after MIDIs and readable/JSON note tables in
  `experiments/research_ab_synthetic_2026-10-05`. Example: one accompaniment pitch
  40 -> 43; all protected 80s and all nonpitch fields unchanged. Arithmetic
  synthetic objective is a test double, not an added scientific style metric.
- Four correctness levels: operator/boundary/rejection unit checks; 200-example
  full-budget protected-event property; SMF0/1, zero-duration/off-as-on-zero,
  duplicates/ties, overlap, channels/tracks/controllers/unmatched-event round trips;
  exact raw-field differential plus original identity and historical E3 transposition.
- Synthetic end-to-end A/B workflow: fit persistence/reload equality before score,
  matched train-only imputation/scaling/selection, class/sample/work/hash leakage,
  source-path/hash failures, reviewed protocol barriers, identity movement, no test
  source reads and all 48 future dispatch budgets/seeds. The search callbacks in
  this workflow test are synthetic doubles, not real scientific optimizations.

Final command results and preservation audit are recorded below after checks.

## Risks and review boundary

Skyline remains an operational approximation; FIFO and overlapping MIDI voices
remain ambiguous. Exact raw event preservation does not prove harmony, phrasing,
voice-leading or musicological melody recognition. Very limited edit budgets,
constraints and local optima may yield identity; real B runtime/useful nonidentity
coverage is unmeasured. Parsing supports the existing SMF0/1 contract.

jSymbolic schema/repeatability proves integration, not composer discrimination or
validity on edited music. The 42-feature engineering subset is not accuracy-tuned.
The retained tempo-standardized beat histogram is velocity-weighted and can still
carry rubato/dynamic dependence; excluding direct expressive fields does not make
all descriptors performance-invariant. Logistic93 is circular when optimized,
custom/external representations share corpus, and no perceptual evaluator is
validated here. A real fit may lose many constant/all-missing columns or encounter
coverage/convergence problems. Failures remain in reports and matched denominators.

A external evaluator must be reviewed and frozen before B. B continuation
thresholds and blinded listening design remain **unresolved and intentionally
unfilled**, as required by the decision. The gate also requires exact input,
backend and both fit fingerprints. Neither fitting accuracy nor passing content
checks automatically promotes the candidate.

## Exact later commands (NOT executed)

Working directory `D:/Studia/inzynierka_dev`. Review files already exist locally:
`experiments/research_ab_review_a.json` and `experiments/research_ab_review_b.json`.
Both have `approved_for_execution=false`, actual manifest/split/backend fingerprints;
B has the preserved V2-05 objective-bundle hash. After human review/authorization,
A's flag must be approved. After A review, B additionally needs the NEW A external
bundle SHA256, approved continuation thresholds/listening design and its approval.
The tracked template documents these required fields. Missing/false review fields
cause an error before scientific source reads/search. Running these commands now
will not run the science.

```powershell
.venv/Scripts/python.exe -B tools/research_ab.py --mode evaluate-a --distribution experiments/research_ab_verification_2026-10-05/jsymbolic/jSymbolic_2_2_user --java 'C:\Program Files\Java\jre1.8.0_291\bin\java.exe' --v205 experiments/research_v2_05_pilot_2026-10-05 --reviewed-protocol experiments/research_ab_review_a.json --output experiments/research_ab_a_evaluation_reviewed

.venv/Scripts/python.exe -B tools/research_ab.py --mode pilot-b --distribution experiments/research_ab_verification_2026-10-05/jsymbolic/jSymbolic_2_2_user --java 'C:\Program Files\Java\jre1.8.0_291\bin\java.exe' --v205 experiments/research_v2_05_pilot_2026-10-05 --external-bundle experiments/research_ab_a_evaluation_reviewed/frozen_external.joblib --reviewed-protocol experiments/research_ab_review_b.json --output experiments/research_ab_b_pilot_reviewed
```

A covers the same inner-training/validation works for external/custom93 classifiers
and the 18 existing V2-05 outputs, with failures/matched work summaries. B's
metadata-only prospective cohort is six lexical validation sources, 12 directed
tasks, seeds1729/1730 and two mechanisms = 48 searches; 512 proposals including
identity per search. E3's initial population already contains identity, so
32+16*30=512 proposals; its 545 cached/population evaluation requests are separately
reported. Pitch search has 512 requests including identity. This is a comparison
of complete mechanisms. Prospective identities are saved in
`experiments/research_ab_verification_2026-10-05/future_cohort_plan.json`; their
selection read metadata only and did not inspect original validation MIDI/scores.

## Final test commands and preservation result

Interpreter: existing `.venv/Scripts/python.exe`, Python 3.10.20; no project
package installation/upgrade or GPU environment change. Java came from the
already installed JRE. The official ZIP was downloaded/extracted locally; only
locks/configuration/code/docs/tests are Git changes.

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --durations=5` | **600 passed**, 72 deselected, 14 existing dependency warnings; 29.88 s |
| `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider -o addopts= tests/property --hypothesis-profile=dev` | **18 passed**, 13 existing warnings; 11.23 s |
| `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider -o addopts= tests/unit/test_property_profiles.py tests/unit/test_research_ab.py tests/property/accompaniment/test_pitch_invariants.py --hypothesis-profile=full` | **7 passed**, including final 200-example new property and complete synthetic workflow; 7.33 s |
| `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider -o addopts= tests/unit/test_accompaniment_search.py tests/unit/test_jsymbolic.py tests/unit/test_research_ab.py tests/unit/e3 tests/unit/test_objective_pilot.py tests/unit/test_v2_content_metrics.py tests/unit/test_midi_parser.py tests/property/accompaniment/test_pitch_invariants.py --hypothesis-profile=full` | **128 passed**, including all 44 new cases and the 200-example property; 14 existing warnings; 11.63 s |
| `.venv/Scripts/python.exe -B tools/research_ab.py --mode feasibility --distribution experiments/research_ab_verification_2026-10-05/jsymbolic/jSymbolic_2_2_user --java 'C:\Program Files\Java\jre1.8.0_291\bin\java.exe' --output experiments/research_ab_feasibility_2026-10-05` | **passed=True; 18/18**, exact repeats; only nine permitted training MIDIs |
| `.venv/Scripts/python.exe -B tools/research_ab.py --mode synthetic --output experiments/research_ab_synthetic_2026-10-05` | **passed=True; three repeated synthetic examples**, before/after tables/MIDIs |
| `git diff --check` and independent fingerprint/reverse-import scan | **passed** |

Development failures were retained as diagnostics and fixed: Windows-default
comma-decimal Java export; one intentionally infeasible synthetic overtopping
fixture; the original property-tier inventory expects original top-level
budgets, so the new property uses its own subdirectory and the existing profile
helper (dev20/full200). Existing tests and tier configuration were not changed.
Final results above apply to the completed source.

Before/after audit: **4,446 existing files checked; 4,441 frozen files unchanged**;
only the five explicitly allowed navigation documents changed. Nine accessed
original source hashes still match; zero frozen/source/result/protocol/completion
changes; zero reverse imports from E1-E4; no outer-test originals opened even for
hashing. The original full-corpus cache was not loaded. Audit:
`experiments/research_ab_verification_2026-10-05/verification.json`; initial hash
snapshot: `frozen_before.json`. Historical artifacts were hashed solely for
preservation and were neither loaded for fitting nor used to evaluate B.

Base HEAD remains `b9d88dc250ea3593879c2ec90647fae88135a795` on
`refactor/research-v2`. Sixteen new files plus five navigation updates;
no commit, branch switch, merge, main change or push. Implementation is ready for
review; the scientific continuation remains gated.

A separate CLI gate check refused `--mode pilot-b` without a reviewed protocol
before research imports/data access and created no output directory.
