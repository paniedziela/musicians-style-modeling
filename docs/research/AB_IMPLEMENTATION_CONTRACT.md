# A+B implementation contract (recorded before source changes)

2026-10-05; branch refactor/research-v2. Implementation and verification only.
Authority: committed RESEARCH_V2_DIRECTION_DECISION.md and current user request.
Read original five references at 324052a and current versions, V2-03 completion
(the protocol is embedded in ROADMAP/content contract; no standalone V2-03
protocol exists), V2-04/05 protocols/completions, local L0034 and L0103 PDFs.

| Literature claim | Intended adaptation | Deliberate deviation | Independent verification |
|---|---|---|---|
| L0034 global jSymbolic features support attribution and classifier guidance | jSymbolic 2.2 external evaluator, fixed logistic configuration | Modern release/features, grouped inner partitions, no WEKA LogitBoost, no FuX/counterpoint/VNS reproduction; never optimize jSymbolic | Pinned runtime/config hashes, ACE dimensions/schema, null/failure/repeat tests, train-only poisoning and identity-overlap tests |
| L0103 section 2 local single-note pitch replacement within a major third | Original NoteId to proposed pitch, +/-4 from ORIGINAL pitch | Polyphonic piano accompaniment, immutable source melody, fixed all timing/velocity/channel/count, 20% floor ceiling | Boundary/unit/property tests and exact raw-event differential checks |
| L0103 visited-state memory and local improvement | Seeded bounded strict-improvement search, identity included in 512 requests | Full visited memory, no escaping local optima/Pareto/Markov/counterexamples/chord annotations, no split/join/duration edits, logistic93 DEVELOPMENT affinity only | Same seed result/history, cache/rejection/budget and negative-score identity fallback tests |
| L0034 classifier alone can exploit nonmusical solutions | Keep technical feasibility distinct from musical validity | No added harmonic objective or invented musical threshold | Report limitation; tests establish software contracts only |

The L0103 component is inspired by/adapted from the paper, NOT a reproduction.

## Reuse ledger (required before imports)

- e3.structure.analyse_structure, e3.types.NoteId: source-only FIFO note tuples;
  highest note at each onset and deterministic duplicate ordinal; IDs contain
  original pitch and MUST NOT be recomputed on candidates. Valid as operational
  mask/identity, not voice/musicological truth. Test duplicates, ties, original
  mask under pitch changes and permutation stability on distinct events.
- midi.parser.MidiParser.parse_bytes: per-track FIFO pairing and supported SMF
  0/1, metadata subset. Valid for feature/tuple comparison, not raw-message
  preservation. Test zero-duration/off-as-on-zero, overlap, cross-track/channel.
- e3.objective.validate_constraints with E3Genome() and roundtrip=False: historical
  transpose-relative mask, relaxed length/count and source/target polyphony max.
  Valid only as ADDITIONAL zero-shift feasibility checks. New adapter separately
  enforces exact fields, original IDs, raw-message pairing/serialization and
  overlap/overtopping rules. Zero genome is validator argument, not new genotype.
- content_metrics.observe_midi/measure_content: observable protected event
  inclusion, per-track pairing ambiguity and structural metadata; order remains
  diagnostic. Test altered pitch/velocity/off and raw-message preservation.
- MidiPrettyPrinter: NOT used for B candidate serialization: it drops controllers,
  track placement/descriptive metadata and can reassociate overlap durations.
  Instead patch only paired raw on/off note numbers in copies of original MIDI;
  identity returns original bytes. Compare raw messages independently, reparse
  expected tuples, retain all metadata/controllers and event positions.
- style_metrics.LOGISTIC/assert_disjoint: exact V2-04 classifier settings and
  sample/work/hash leakage guard. A adds explicit train-only median imputation
  for nullable external fields; variance/scaling/model remain the same.
- e1.composition_features.extract_composition_features: frozen 93 contract for B
  objective; use a frozen already-fitted logistic pipeline, no fitting in search.
  Verify equality against direct pipeline probability and parameter snapshot.
- feature_feasibility.select_pilot/sanitize_midi/attempt/compare_attempts:
  lexical nine OUTER-TRAIN samples, neutral name, explicit failures, no full-cache
  audit. Valid only for feasibility; no corpus-cache access. Independent synthetic
  selection/sanitization/failure tests. Real attempts confined to these nine.
- provenance.sha256_file/write_json/collect_provenance and asset_paths roots:
  pure accounting/filesystem provenance. Existing tests retained; hashes and
  fresh-destination guards verified in new workflow.

No E3 proposal/mutation, five-gene genotype, transposition assumption or historical
metric is reused by B search. No reverse import or frozen file change.
Java/sklearn remain CPU; GPU availability/environment is untouched. No compatible
GPU implementation exists for these frozen components, so no algorithm change or
GPU benchmark is planned.

Stop before real A evaluation/B pilot. Later commands must require explicit
reviewed protocol artifacts; missing continuation/listening thresholds remain
an execution gate, not fabricated decisions in this implementation task.

## Final concrete configuration / reuse additions

Track A freezes 42 named upstream descriptors (638 flattened dimensions) before
the nine-source feasibility run. Histogram dimensions came from synthetic ACE
exports, not corpus fitting. Selection is an engineering baseline, not a paper's
111/12-feature reproduction and not an accuracy-tuned subset. It covers absolute
pitch/classes, melodic and vertical intervals/chords, meter/rhythm/rest and
texture/density. No file/label features, instrumentation/program fields,
direct velocity descriptors, expressive pitch-bend/vibrato/microtone/glissando or absolute tempo/
seconds features. Constant remaining fields are removed by training variance.
Median imputation is a deliberate new preprocessing step for nullable external
values; all-missing train columns map to zero and are variance-excluded. No
imputation or classifier is fitted on the real nine-source feasibility set.

Explicit JVM English/US locale prevents the observed comma-decimal ACE export
failure under the Windows default Polish locale. The configuration requests
whole-piece extraction, XML and CSV, no ARFF/windowing. The Java heap ceiling is
2 GiB, a resource choice rather than a scientific parameter; executable/version/
SHA and all distribution JAR/config/schema hashes are recorded. No environment
variable or GPU setting is changed.

For the later E3 comparison only, `objective_pilot.PilotObjective/run_search`
reuse the independently tested V2-05 private zero-transpose bindings, serialized
logistic93 objective and protected-content checks. E3 proposal/mutation code is
used exclusively in this named comparator, never in the new pitch-search
mechanism. `e3.algorithm.SearchConfig` uses population32/generations16/
stagnation17, inherited elite2/tournament3/sigmas. Identity is already in the
initial population: 32 + 16*30 = 512 proposals INCLUDING identity; 545 evaluation
requests include retained populations and the separately cached identity request.
Pitch search has 512 proposals/requests including identity. Synthetic workflow
and retained V2-05 tests verify dispatch, frozen globals, content and budgets.
No real comparison or pilot is executed here.

`style_audit.validate_splits/cluster_summary/agreement/movement` and
`style_metrics.rank_credit` are reused by later reporting. Their existing tests
plus synthetic workflow tests independently cover group/hash leakage, matched
failure denominators, identity movement and serialization of frozen fits. Their
conditional work bootstrap remains descriptive; no new style metric is added.

The retained upstream tempo-standardized beat histogram is still velocity-weighted
and can retain rubato/dynamic effects. Direct expressive descriptors are excluded,
but this residual performance dependence is a named evaluator risk; no claim of
complete performance invariance or musical validity is made. Feature semantics
were checked against the pinned synthetic ACE descriptions.
