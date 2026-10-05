# RESEARCH_CONTEXT — Research V2

Updated 2026-10-05. V2-05 fixed-budget objective pilot is complete: 18/18 runs completed; execution/preservation passed=True. Stop after V2-05 for review; no full E1c, outer-test evaluation, larger transfer experiment or push. V2-01 through V2-04 remain preserved. See [V2_05_COMPLETION.md](V2_05_COMPLETION.md) and [V2_05_PROTOCOL.md](V2_05_PROTOCOL.md).

## Research question and evidence

Transform a source symbolic score toward a corpus-defined target composer while preserving explicitly protected melody. Generation from scratch, expressive performance modelling and arbitrary-composer few-shot transfer are secondary.

The local audit finds 150 accepted score MIDIs / 87 work groups (Bach 59/30, Beethoven 57/28, Chopin 34/29); each has matching MusicXML. The physical ASAP folder has 235 score MIDI/XML pairs. The custom93 implementation reproduces its frozen feature cache in the preceding read-only audit; V2-01 checks its snapshot/hash consistency without extracting features again.

E1b RF balanced accuracy is 0.86424, clustered CI [0.83781, 0.89059], macro-F1 0.86956 and retrained permutation p=0.01099. This shows composer-discriminative signal within this corpus, not a universal definition of style. E2's mean target-probability movement is 0.02767; E3's is 0.03771, with paired E3−E2 improvement 0.01004. E3 objective/evaluator Pearson agreement is only 0.04395. These are reported experimental statistics, not new V2 computations. E4.6 fails its validation gate; keep it as a negative result rather than repeatedly tuning GAN variants.

## Content contract

V2-03 review clarification recorded before V2-02 implementation (2026-10-04): ordering of distinct simultaneous same-tick MIDI events remains available as a diagnostic, but does not by itself constitute a hard melodic-identity violation. Exact protected pitch, onset and note-off / identifiable duration are the primary hard requirements. Completed V2-03 code, scientific artifacts and completion statistics retain their historical strict-order interpretation; they are not rewritten or regenerated.

Content and style are separate axes. Version each experiment's melody selector and constraints; a Skyline mask is an operational approximation, not a musicological ground truth.

| Attribute | Policy | Interpretation |
|---|---|---|
| Protected melody pitch | PRESERVE exactly | Original absolute pitches; no global transposition in new transfer experiments |
| Protected onset and note-off / identifiable duration | PRESERVE exactly | Primary hard melodic identity requirements alongside protected absolute pitch |
| Ordering of distinct simultaneous same-tick MIDI events | Diagnostic | Remains measurable, but alone is not a hard violation of melodic identity |
| Meter, essential metadata, MIDI format/resolution and channels where required | PRESERVE | Structural/technical invariants, distinct from melodic identity |
| Velocity when the transformation leaves it unchanged | Report separately | Current transformation invariant; score-MIDI velocity is not automatically fundamental melodic content |
| Harmonic/chroma relation, phrase/section structure | SOFTLY_PRESERVE | Named measurements and predeclared tolerances; no invented aggregate content scalar |
| Accompaniment pitch, timing, duration, voicing/texture and bounded density | STYLE_TRANSFORMABLE | Subject to protected events, extent and validity budgets |

Historical E3 allowed global transposition up to ±6 semitones while preserving its selected melody relative to that transposition. Preserve that baseline and its reported interpretation. The new exact-pitch V2 policy does not retroactively make E3 a failed exact-pitch experiment. An optional transposition-policy ablation may be proposed later.

Overlapping same-pitch MIDI events can make parsed note tuples ambiguous. V2-03 compares the original protected on/off events with multiplicity and observable order, while marking unidentifiable durations and simultaneous occurrence/order as ambiguous. Nineteen reselected-output Skyline discrepancies comprise 17 pairing/reselection cases and two higher-note cases; the original protected tuples remain present under historical transposition. Unmatched raw events remain explicit undefined cases even if the frozen parser drops them. Historical E3 permits same-tick permutations; the completed V2-03 strict literal-order assessment is retained as a diagnostic under the review clarification above. Note-count, polyphony and piece duration are validity/structure measurements, not substitutes for melody identity.

## Style definitions and evaluator dependence

Style is a corpus-relative pattern of pitch/register, rhythm/metrical placement, harmony, texture, repetition and accompaniment. Compare competing definitions before choosing an optimization objective.

The E1b RF is a **separate held-out evaluator** of E3 output, fitted on fold training works. It is not necessarily a fully independent definition of style: the corpus and overlapping feature families create dependence even when the fitted model differs. If a classifier becomes an objective, separate its fitting/evaluation roles and report remaining corpus/representation dependence explicitly. A distinct event-profile measurement (L0074) is useful precisely because it changes the representation and measurement structure; even that does not establish perceptual validity alone.

V2-04 completed the fixed E1d comparison: RMS67, Gaussian67, logistic93, onset-duration cosine and time-pitch cosine, all fit only on matching outer train partitions. No musif/full E1c, metric combination or model selection.

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

Gaussian uses the recorded train-only floor `max(target std, 0.05 * pooled training std, 1e-6)`; RMS retains its frozen std replacement. Logistic C=1/balanced/train-only variance/scaling/max_iter=5000 are recorded settings, with lbfgs an implementation choice. The duration-to-velocity reassociation synthetic counterexample shows why serialized RMS fallback movement can differ while custom93/all-note event profiles stay unchanged. All correlations/undefined cases and six directions remain in `style_verified/summary.json`. StyleRank/sequential extensions and larger transfer experiments remain separate future decisions.

## Leakage-safe representation research

E1c reuses the frozen 5 repeats × 5 outer grouped folds with 3 inner folds. Fit imputation, scaling, selection, thresholds, profiles, classifiers and embedding/ranking models within the appropriate training fold. Keep names, metadata, paths and composer labels out of numerical features. Compare custom93, musif, jSymbolic, selected music21 and schema-defined combinations where feasible. Preserve custom93's names/order/semantics; wrappers may import frozen E1.

The first feature pilot is nine predeclared samples, not two successful extractions. Select one lexical sample from each of the first three lexical work groups per composer within repeat 0 / outer fold 0 training data. Every sample gets success or an explicit failure record in both deterministic repetitions. Report schemas, versions and failure-status stability. V2-02 completed all 18 attempts per backend with exact repeated schemas/values/outcomes and a separate exact 150-sample custom93/cache audit. Full custom93/cache comparisons are scientific audit artifacts, not default unit tests. musif retains 203-329 score-level numeric features per sample (405 in the descriptive union); nullable tempo and varying interval vocabulary prevent treating the pilot as a ready classifier input. No imputation, selection or classification was fitted.

Matched MusicXML warrants a separate score-aware E1m investigation with musif/Partitura: notation may add information absent from score MIDI. Keep sample/work identities and matched-subset comparisons; do not silently merge symbolic-score and performance data. musif 1.2.4/music21 9.9.2 are installed only in the separate V2-02 Python 3.10 environment. They remain absent from the frozen project `.venv`; Partitura and a jSymbolic JAR remain unavailable. The tracked isolated dependency lock and actual worker versions accompany the V2-02 record.

## Architecture and boundaries

Reuse current representation, extraction and search where appropriate. V2-01 adds `asset_paths.py`, `provenance.py` and `research_audit.py`; V2-03 adds only `content_metrics.py` and the explicit `content_audit.py` runner. V2-02 adds concrete custom93/musif `feature_backends` and the explicit `feature_feasibility.py` runner only. `style_metrics.py` and `style_audit.py` now serve only concrete V2-04 needs, with no registry or reverse imports. No `musicians_style.v2`, duplicated E1–E4 implementation, speculative refactor or reverse dependency from frozen packages.

The explicit user request authorized the completed bounded V2-05 after E1d review. Prefer an external adapter/runner around frozen E3 to enforce zero transposition and exact protected pitches. Change shared E3 code only if external enforcement proves insufficient, with a separately reviewed minimal patch, unchanged old default and regression coverage. The fixed V2-05 set is RMS67/Gaussian67/logistic93; no larger experiment is authorized.

## Evidence and test policy

Historical plans/reports remain unchanged; STATUS.md, this context, the matrix, registry and roadmap are the current entry points. Local results and checkpoints are ignored by Git but remain evidence. Record unavailable or contradictory historical provenance instead of fabricating a historical code/environment snapshot.

Default pytest runs fast unit/smoke checks against checkout source. Property tests use explicit development/full profiles; integration/regression workflows are explicit. Dataset-wide scientific checks produce fresh audit artifacts. V2-01 remains infrastructure verification; V2-03 adds content measurements only. V2-02 performs only the nine-sample feature pilot and explicit custom93 equivalence audit. V2-04 subsequently fitted only fixed fold-local style measures; no full E1c selection, transfer generation or change to frozen E1–E4/V2-01–03 protocols was performed.

## V2-05 exploratory result

| Objective | Exact observable policy | time_pitch positive / negative / null | time_pitch mean | Optimized mean |
|---|---:|---|---:|---:|
| rms67 | 6/6 | 1 / 5 / 0 | -0.022067 | 0.377232 |
| gaussian67 | 6/6 | 2 / 4 / 0 | -0.026364 | 0.026629 |
| logistic93 | 6/6 | 0 / 6 / 0 | -0.068912 | 0.139354 |

Definitive artifacts: `experiments/research_v2_05_pilot_2026-10-05`. All comparisons are exploratory, with content ambiguity/evaluator dependence recorded separately. Optimized gains use different scales and cannot rank objectives across metrics. Stop for V2-05 review.
