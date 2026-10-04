# RESEARCH_CONTEXT — Research V2

Updated 2026-10-04. Research V2 is a research/experiment phase within the current implementation tree. E1–E4 are frozen evidence, not the final architecture. V2-01, V2-03 and the separately authorized V2-02 feature feasibility are complete. Stop for V2-02 review; full E1c/V2-04/V2-05 remain unstarted. See [V2_02_COMPLETION.md](V2_02_COMPLETION.md) for feature findings and limitations. See [V2_03_COMPLETION.md](V2_03_COMPLETION.md) for the content contract, measurements and verification.

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

Historical E3 allowed global transposition up to ±6 semitones while preserving its selected melody relative to that transposition. Preserve that baseline and its reported interpretation. The future exact-pitch V2 policy does not retroactively make E3 a failed exact-pitch experiment. An optional transposition-policy ablation may be proposed later.

Overlapping same-pitch MIDI events can make parsed note tuples ambiguous. V2-03 compares the original protected on/off events with multiplicity and observable order, while marking unidentifiable durations and simultaneous occurrence/order as ambiguous. Nineteen reselected-output Skyline discrepancies comprise 17 pairing/reselection cases and two higher-note cases; the original protected tuples remain present under historical transposition. Unmatched raw events remain explicit undefined cases even if the frozen parser drops them. Historical E3 permits same-tick permutations; the completed V2-03 strict literal-order assessment is retained as a diagnostic under the review clarification above. Note-count, polyphony and piece duration are validity/structure measurements, not substitutes for melody identity.

## Style definitions and evaluator dependence

Style is a corpus-relative pattern of pitch/register, rhythm/metrical placement, harmony, texture, repetition and accompaniment. Compare competing definitions before choosing an optimization objective.

The E1b RF is a **separate held-out evaluator** of E3 output, fitted on fold training works. It is not necessarily a fully independent definition of style: the corpus and overlapping feature families create dependence even when the fitted model differs. If a classifier becomes an objective, separate its fitting/evaluation roles and report remaining corpus/representation dependence explicitly. A distinct event-profile measurement (L0074) is useful precisely because it changes the representation and measurement structure; even that does not establish perceptual validity alone.

E1d should compare current standardized RMS profiles, Gaussian profiles, train-only classifier affinity and event profiles on real held-out works, identities and frozen E2/E3 outputs. Relative/StyleRank/sequential measures are bounded extensions. Do not arbitrarily average all scores. Report target ranking, direction effects, degeneracy and agreement alongside content. Failed metrics remain diagnostic evidence and need not consume transfer optimization compute.

## Leakage-safe representation research

E1c reuses the frozen 5 repeats × 5 outer grouped folds with 3 inner folds. Fit imputation, scaling, selection, thresholds, profiles, classifiers and embedding/ranking models within the appropriate training fold. Keep names, metadata, paths and composer labels out of numerical features. Compare custom93, musif, jSymbolic, selected music21 and schema-defined combinations where feasible. Preserve custom93's names/order/semantics; wrappers may import frozen E1.

The first feature pilot is nine predeclared samples, not two successful extractions. Select one lexical sample from each of the first three lexical work groups per composer within repeat 0 / outer fold 0 training data. Every sample gets success or an explicit failure record in both deterministic repetitions. Report schemas, versions and failure-status stability. V2-02 completed all 18 attempts per backend with exact repeated schemas/values/outcomes and a separate exact 150-sample custom93/cache audit. Full custom93/cache comparisons are scientific audit artifacts, not default unit tests. musif retains 203-329 score-level numeric features per sample (405 in the descriptive union); nullable tempo and varying interval vocabulary prevent treating the pilot as a ready classifier input. No imputation, selection or classification was fitted.

Matched MusicXML warrants a separate score-aware E1m investigation with musif/Partitura: notation may add information absent from score MIDI. Keep sample/work identities and matched-subset comparisons; do not silently merge symbolic-score and performance data. musif 1.2.4/music21 9.9.2 are installed only in the separate V2-02 Python 3.10 environment. They remain absent from the frozen project `.venv`; Partitura and a jSymbolic JAR remain unavailable. The tracked isolated dependency lock and actual worker versions accompany the V2-02 record.

## Architecture and boundaries

Reuse current representation, extraction and search where appropriate. V2-01 adds `asset_paths.py`, `provenance.py` and `research_audit.py`; V2-03 adds only `content_metrics.py` and the explicit `content_audit.py` runner. V2-02 adds concrete custom93/musif `feature_backends` and the explicit `feature_feasibility.py` runner only. `style_metrics` remains reserved for separately authorized V2-04. No `musicians_style.v2`, duplicated E1–E4 implementation, speculative refactor or reverse dependency from frozen packages.

Future V2-05 starts only after E1d completion and another review. Prefer an external adapter/runner around frozen E3 to enforce zero transposition and exact protected pitches. Change shared E3 code only if external enforcement proves insufficient, with a separately reviewed minimal patch, unchanged old default and regression coverage. No objective candidates are selected for transfer now.

## Evidence and test policy

Historical plans/reports remain unchanged; STATUS.md, this context, the matrix, registry and roadmap are the current entry points. Local results and checkpoints are ignored by Git but remain evidence. Record unavailable or contradictory historical provenance instead of fabricating a historical code/environment snapshot.

Default pytest runs fast unit/smoke checks against checkout source. Property tests use explicit development/full profiles; integration/regression workflows are explicit. Dataset-wide scientific checks produce fresh audit artifacts. V2-01 remains infrastructure verification; V2-03 adds content measurements only. V2-02 performs only the nine-sample feature pilot and explicit custom93 equivalence audit. No model fitting, full E1c, transfer generation or change to frozen protocols was performed.
