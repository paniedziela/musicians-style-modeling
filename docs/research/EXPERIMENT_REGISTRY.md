# EXPERIMENT_REGISTRY — experiment identities and evidence

The registry separates historical/frozen scientific artifacts from new Research V2 experiments.

## Frozen experiments

| ID | Status | Question | Primary output | Mutability |
|---|---|---|---|---|
| E1 | CLOSED / GO | Do symbolic features distinguish Bach, Beethoven and Chopin under grouped evaluation? | grouped OOF classification results, feature importance | frozen |
| E2 | CLOSED / BASELINE | What does the legacy global GA achieve and damage? | 300 paired transfer tasks | frozen |
| E3 | CLOSED / GO WITH LIMITATIONS | Can constrained local GA improve target-style proxy while protecting content better than E2? | 300 paired tasks + content/style metrics | frozen |
| E4.5 | CLOSED / NO-GO | First conditional GAN protocol | validation-only negative result | frozen |
| E4.6 | CLOSED / NO-GO | Improved conditional GAN protocol | validation-only negative result; outer test unopened | frozen |

No frozen experiment is silently rerun under the same schema after changing code.

## Research V2 proposed experiments

### E1c — feature extractor comparison

**Hypothesis**

Established MIR descriptors and the thesis-specific 93-feature representation contain complementary composer-discriminative information.

**Variants**
- custom93;
- musif;
- jSymbolic;
- selected music21;
- custom93 + musif;
- custom93 + jSymbolic;
- selected combined representation.

**Protocol**

Reuse the E1 grouped split logic. Feature selection/scaling is train-only.

**Primary metrics**
- balanced accuracy;
- macro-F1;
- clustered confidence interval;
- runtime and feature count;
- feature-family importance/stability.

**Acceptance**

A technically valid result is sufficient. An external extractor does not need to beat custom93 to be scientifically useful.

**Artifacts**
- feature schema per backend;
- cached feature matrix;
- fold predictions;
- comparison report.

### E1d — style metric / representation comparison

**Hypothesis**

Different corpus-relative style measures produce meaningfully different rankings, and some agree better with held-out composer identity than the current E3 objective.

**Candidate metrics**
- current standardized profile distance;
- classifier probability;
- Gaussian profile score;
- target-vs-counterexample score;
- StyleRank-like score;
- Groove2Groove-inspired event profiles;
- optional Markov/pattern score.

**Data**

Evaluate both real held-out works and frozen E2/E3 outputs.

**Primary analyses**
- target-vs-source ranking on real works;
- correlation/agreement among style metrics;
- relation to E3 internal objective;
- direction-wise behavior.

**Acceptance**

At least two independent style measures implemented and compared.

### E3a — classifier-driven objective

Use train-only composer probability/log-odds as optimization signal.

Critical requirement:
evaluation uses a separately fitted or cross-fitted evaluator to avoid circular evidence.

### E3b — distribution-driven objective

Use Gaussian/empirical feature distributions inspired by L0067.

Ablate semantic families such as pitch, rhythm, harmony and texture.

### E3c — relative / sequential objective

Candidates:
- target-vs-counterexample classifier/log-likelihood;
- Markov transition score;
- small pattern model inspired by L0027/L0103.

Implement only after E3a/E3b provide a stable comparison harness.

### E3x — objective comparison

Use one common transformation harness and equal evaluation budget.

Report:

```text
objective
→ internal objective gain
→ independent target-style gain
→ content preservation
→ runtime
→ failure / identity rate
```

The main question is objective validity, not which optimizer achieves the largest uncalibrated fitness.

### E1m — optional MusicXML / score-aware comparison

Use matching ASAP works:
- MIDI/custom or MIDI/musif;
- MusicXML/musif or Partitura-derived features.

Question:
does richer notation add composer-discriminative information beyond score MIDI?

### E5 — optional neural comparison

This is intentionally not named E4.7.

Candidates should be reassessed after E1c/E1d/E3x:
- VAE / Transformer Autoencoder;
- MuseMorphose-inspired conditional model;
- METEOR-inspired melody-aware model;
- public pretrained composer model adaptation;
- Composer Vector if reproduction is bounded.

**GO criterion**

The selected model answers a research question not already answered by the interpretable pipeline and uses the same content/style evaluation framework.

## Shared output contract for new V2 experiments

Each run should record:

```text
run_manifest.json
config_snapshot.yaml
git.json
environment.json
dataset_fingerprint.json
split_fingerprint.json
feature_schema.json
objective_schema.json
progress.jsonl
results.json
report.md
```

Where applicable:
- predictions;
- transformed MIDI;
- cached features;
- fitted train-only profiles/models;
- plots.

## Naming policy

Use explicit immutable identifiers, for example:

```text
custom93_v1
musif_v1
jsymbolic_v1
style_metric_classifier_v1
style_metric_gaussian_v1
style_metric_stylerank_v1
e3_objective_classifier_v1
```

Never reuse an identifier after altering semantics.
