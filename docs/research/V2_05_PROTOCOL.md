# V2-05 fixed-budget objective pilot — 2026-10-05

The user authorized V2-05 after V2-04 and fixed RMS67, Gaussian67 and logistic93. This supersedes provisional two-candidate language and the V2-04 stop boundary only for this pilot. No broad replanning, E1c, StyleRank, sequence/neural models, new operators, outer-test evaluation or 300-task experiment.

## Cohort and fitting

Use frozen repeat 0 / outer fold 0 / inner fold 0. Select first lexical (group_id, sample_id) validation sample per composer, before extraction or inspection of style scores, transfer or note count. Six directed pairs x three objectives = 18 real searches. Persist cohort, tasks, manifest/split/source hashes, objective/evaluation contracts and search budget before search. Outer-test original MIDI/features are never loaded. Identity metadata are used for leakage checks. Previous scientific artifacts are byte-hashed solely for preservation.

Fit all profiles, Gaussian scales, event prototypes, logistic variance/scaling/model and historical-configuration RF on the same inner-training partition. All other identities, including validation and outer test, are forbidden by sample/group/SHA. Reuse V2-04 formulas/configurations unchanged; no outer-fold models or feature table exposing outer-test values. RF uses frozen E3 parameters (300 trees, balanced, max_features=0.5, min_samples_leaf=2, random_state=1729, n_jobs=1), refitted on inner training. Persist joblib and detailed learned provenance/hashes before optimization. No fitting, tuning, selection or calibration on pilot outputs.

## Search and content

User-fixed seed 1729, population 32, 60 generations, stagnation 61. User confirmed inherited elitism 2, tournament 3, mutation sigma [1.0,0.12,0.12,0.12,0.12]. First sigma remains an inactive historical transpose draw: canonicalize transpose to zero before transformation/evaluation/cache access. Retain all five random draws and frozen GA dynamics/operators. Use original E3 function code with private globals binding for canonicalization; never mutate shared globals or frozen E3 files. External adapter requires no frozen-source hook.

Per-run budget: initial 32 plus 60x30 new offspring = 1,832 proposals; identity plus 61x32 population requests = 1,953 evaluation requests (including retained elites/cache hits). Record unique transformed-representation evaluations, transformations, cache hits, inactive-gene canonicalizations, all 61 histories, rejection reasons, stop reason and runtime. Stagnation cannot stop within 60 generations.

Protect original Skyline absolute pitch, onset, note-off/identifiable duration and velocity. Use frozen E3 feasibility and reused V2-03 raw serialized content/structure checks. Same-tick order is diagnostic. Observable identity must pass; duration failed/undefined is infeasible; overlapping duration ambiguity is reported, never labelled certified exact. Structural invariants must pass. Identity outputs retain original bytes. Other candidates use deterministic existing serialization; optimize serialized representation, avoiding V2-04 identity serialization/reselection discrepancy. Gain = output target affinity minus original-source target affinity. Identity gain is zero. No composite style/content score.

## Evaluation and review

Every output: optimized movement, all V2-04 style scores/movement/source drop/target-source margin, primary structurally distinct time_pitch, secondary onset_duration, logistic93 explicitly non-independent when optimized, historical-configuration RF with shared-corpus/custom93 dependence, all V2-03 original-mask content/ambiguity/order/velocity/structural diagnostics, E3 constraints and search accounting/history/runtime. Source-self gives zero identity references through input scores; byte-identical outputs counted. Frozen E3 remains historical transposition-allowed evidence, never regenerated/reinterpreted as this pilot.

Use V2-04 null/negative convention 1e-12. Compare content, distinct movement, six-direction consistency, negative/null/undefined rates and optimized/non-optimized disagreement separately. RMS/Gaussian share 67 features; both link and variance differ. RF/logistic share corpus/custom93. Event profiles share parser/corpus but differ structurally; no perceptual independence claim.

Three fixed sources/works, exploratory feasibility only. Descriptive means and individual directions; no inferential p-values or confirmatory intervals. Repeated synthetic same-seed search tests for all objectives, persisted-fit score equivalence and repeat output serialization check reproducibility without real runs beyond 18. Real optimization reruns remain unmeasured. No automatic objective promotion, larger experiment or push. Stop after V2-05 for review.
