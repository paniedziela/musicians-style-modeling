# EXPERIMENT_REGISTRY — frozen evidence and proposed Research V2 experiments

Updated 2026-10-04. Experiment identifiers describe scientific protocols; Research V2 does not create a parallel Python implementation tree. V2-01, V2-03 and the separately authorized V2-02 feature feasibility are complete. Stop for V2-02 review; full E1c/V2-04/V2-05 remain unstarted. No frozen evidence is rewritten or silently rerun under an existing schema.

## Frozen identities

| ID / schema | Local artifacts | Evidence and status |
|---|---|---|
| E1 / e1.0.0 manifest, e1.1.0 splits, e1.3.0 composition features | `datasets/derived/e1_asap`; `experiments/e1_asap_e1b_2026-09-02_005705_449734` | CLOSED / GO; 150 samples, 87 works; grouped 5×5 outer / 3 inner; custom93; RF BA 0.86424 |
| E2 / e2.0.0 | `experiments/e2_asap`; `configs/e2_asap.yaml` | CLOSED / FROZEN; 300 outputs; global four-parameter legacy GA, 42-feature objective / custom93 held-out RF evaluator; Δp 0.02767 |
| E3 / historical e3 artifacts | `experiments/e3_asap`; `configs/e3_asap.yaml` | CLOSED / GO WITH LIMITATIONS; 300 outputs; local GA with 67 event-profile components; RMS-z objective; Δp 0.03771; objective/evaluator r=0.04395 |
| E4.5 / historical v2 artifacts | `experiments/e4_asap_v2` | CLOSED / validation NO-GO; retain earlier conditional GAN negative result |
| E4.6 / e4.6.0 status | `experiments/e4_asap_v3`; `configs/e4_asap_v3.yaml` | CLOSED / validation NO-GO; 86 validation MIDIs; best.pt SHA256 e165ded23e0202dd6798dbad6ff16f5ca33fce8c6c13428983412e1fdf0c0d5d |

E2/E3 use repeat 0 paired directions/tasks. E3 mean Δp CI [0.02850, 0.04604]; E3−E2 CI [0.00134, 0.02240], p=0.0395. E3 has 185 positive, 67 zero and 48 negative movements; 45 identity fallbacks. E4.6 mean validation Δp=0.00340, three of six directions positive, melody median 0.91625, onset 0.8540; four validation criteria fail. Local absence of outer-test artifacts agrees with the closed-test policy but cannot prove data was never inspected.

Historical E3's melody policy permits global ±6-semitone transposition. 233/300 outputs use nonzero transposition. Keep it as the transposition-allowed baseline. New transfer experiments require exact original protected melody pitches, onsets and note-offs / identifiable durations. The V2-03 review clarification makes distinct simultaneous same-tick order diagnostic, not a hard identity violation on its own; completed V2-03 artifacts remain unchanged. A later optional policy ablation requires a new identity/config; do not relabel old outputs.

## Historical provenance limitations

Canonical manifest code reference: `08e4cc6a00d6c6230560907455eb246b97f174b7`. E1 run manifest: `fae61` prefix; the historical E1 report references `5e827` prefix. E2 manifest: `9f7177e23230ae59f84703d28a1d2cd67242e92b`. E3 lacks a complete recorded code/environment snapshot. References and anonymized/reworked history are retained as evidence, not reconciled by guessing. Hash/snapshot checks establish current consistency, not full reproducibility of the original environment. V2 audits record their own current checkout HEAD separately.

## V2-01 — evidence inventory and environment guard

Infrastructure audit, not a new musical experiment. Separate `inventory` from explicit `baseline` scope. Reserve a new output directory; record checkout/import/environment/Git/config/root provenance, asset inventory, baseline checks, test-tier inventory and report. Reject ambient imports outside the active checkout's resolved `src/musicians_style`, including `.venv` copies inside the checkout. No automatic repair. Audit failures and unavailable information remain inspectable. Scientific checks remain outside pytest.

## E1c — MIR representation comparison (proposed)

**Hypothesis:** established MIR descriptors may complement custom93. Compare custom93, musif, jSymbolic, selected music21 and named/schema-defined combinations where extraction is feasible. Reuse frozen grouped splits; no split regeneration to accommodate failures. All preprocessing/model choice uses inner training data only. Primary outcomes: grouped balanced accuracy/macro-F1, clustered CIs, family stability, extraction coverage, schema size and measured runtime.

V2-02 begins with nine predeclared training samples and two extraction attempts per sample, including deterministic failure records. Full E1c comes after feasibility review. If an extractor has incomplete coverage, report failures and a matched-subset custom93 comparison; do not compare differently composed populations as if matched. An external extractor need not outperform custom93 for a valid result. Artifacts: pilot manifest, versions/config/source hashes, schemas/cache, failures, fold predictions and report. Frozen outer folds have already been inspected historically; describe further exploration honestly rather than claiming a fresh untouched confirmatory test.

## V2-02 — MIDI feature feasibility (complete; stop for review)

Schema `v2-02.feasibility.1`; fixed nine training samples, two repetitions per backend, 36 records including explicit success/failure status. custom93 imports frozen E1b and exactly reproduces all 150 cache rows in the separate audit. Isolated musif 1.2.4 yields repeat-identical nullable numeric results for 9/9 samples: 203-329 features per row, a 405-feature pilot union, with NumericTempo missing throughout. Names/labels/descriptive MIDI metadata and identifiers are excluded from model features; part/sound/family headers and musical categories are also excluded by this bounded score-level numeric adapter. See [V2_02_COMPLETION.md](V2_02_COMPLETION.md), `experiments/research_v2_02_2026-10-04/pilot_final` and `requirements/research-v2-02-musif.lock.txt`.

This establishes bounded extraction feasibility, not composer discrimination or model readiness. Sample-dependent vocabulary, missing tempo, constant pilot columns and music21 MIDI interpretation remain explicit limitations. No E1c fold predictions/training, MusicXML replacement, jSymbolic/Partitura backend, V2-04 or V2-05.

## E1m — matched MusicXML study (COULD, proposed separately)

Compare MIDI and score-aware extraction on matched 150 identities/work groups with the same grouped protocol. musif/Partitura dependencies and notation-derived features are versioned separately. GO requires usable notation and bounded cost; missing XML and differences in coverage are explicit. No automatic launch during V2-01.

## Content audit — V2-03 (complete; artifacts preserved)

Audit the 600 frozen E2/E3 outputs without generation. Separate protected melody identity, structural/technical invariants, velocity invariance of the current transform, soft harmonic/structure preservation and transformable accompaniment. Compare semantic on/off masks as well as parsed notes for overlaps. Report historical E3 transposition explicitly; new policy is exact-pitch. Produce per-output diagnostics/hashes and direction/cluster summaries outside pytest. Synthetic identity, deliberate corruption, overlap and transposition cases belong in small tests.

Completed with all 600 frozen outputs and 150 source-self identity references under `v2-03.content.1`. See [V2_03_COMPLETION.md](V2_03_COMPLETION.md) and `experiments/research_v2_03_2026-10-04/content_verified`. The original source mask is protected; reselected output Skyline is diagnostic only. Observable on/off retention, identifiable durations/order, structural/technical invariants and protected velocity are separately reported. Soft harmonic/phrase tolerances and an aggregate content scalar were not invented. Existing test tiers and frozen E1-E4 imports remain unchanged.

## E1d — competing style measurements (V2-04, proposed)

Compare named train-only RMS67, Gaussian67, classifier93 and structurally distinct event-profile candidates; this is a candidate evaluation set, not the future transfer pilot selection. Optional relative/StyleRank/sequential extensions require bounded feasibility. Frozen folds govern fitting, scaling, sparse-vocabulary construction, thresholds and target/counterexample populations. Score real held-out works, identities and frozen E2/E3 outputs. Report target ranking, direction effects, score variance, objective/evaluator agreement and content dependence, clustered by original work and paired where appropriate.

RF is a separate held-out evaluator, not necessarily an independent style definition because its corpus/features overlap with candidate measures. Event profiles add a more distinct measurement perspective. Do not use one fitted classifier as objective and sole evaluator.

Predeclare objective eligibility before scoring: leakage-safe fitting; finite, nondegenerate scores; deterministic behavior; grouped held-out target-ranking balanced accuracy with clustered 95% CI lower bound above chance (1/3 for three composers). Fix ranking/tie/CI procedures in the E1d protocol before running it. Failed candidates remain diagnostic evidence, not optimization targets. These criteria do not prove perceptual validity.

## Objective pilot — V2-05 (SHOULD, provisional)

Requires E1d completion and another review. Select up to two eligible objective variants from E1d under the predeclared criteria; no RMS67/Gaussian67/classifier93 final set is hard-coded now. If none qualifies, NO-GO without transfer compute. Freeze selection before transformed-output optimization.

Bound a six-direction pilot, declare source works/seeds/budget/constraints and measure per-task cost before launch. Reuse E3 externally, reject nonzero transposition, check exact protected pitches and report rejected candidates/feasibility budget. Preserve identity and frozen E2/E3 paired baselines. Modify shared E3 only if external enforcement is insufficient, after separate minimal-change review with old-default regression tests. Report style and content separately, remaining evaluator dependence, runtime and failures. No optional policy ablation or neural training is automatic.

## Later decisions

A listening study requires a separate design/approval; an objective win alone is not perceptual success. Keep new GAN tuning, from-scratch Transformers and performance-style modelling deferred. Neural baselines require compatible representation, accessible implementation/checkpoints, bounded cost and a stronger common evaluation framework.
