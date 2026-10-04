# PAPER_IMPLEMENTATION_MATRIX — selected thesis literature

This matrix turns the selected `L*.pdf` literature into implementation decisions.
Priority means usefulness for this repository, not general paper quality.

| ID | Paper | Core idea | Best use here | Action | Priority |
|---|---|---|---|---|---|
| L0027 | **Automatic Modeling of Musical Style** — Lartillot, Dubnov, Assayag, Bejerano (2001) | Unsupervised sequence/pattern modelling using Incremental Parsing/compression ideas and PST | A genuinely different sequential/pattern definition of style; historical bridge to OpenMusic/LZ | `ADAPT_METHOD`, `RELATED_WORK` | STRONG ADDITION |
| L0034 | **Classification and Generation of Composer-Specific Music Using Global Feature Models and Variable Neighborhood Search** — Herremans, Sörensen, Martens (2015) | Global features → composer classifier → composer probability inside VNS | Direct methodological predecessor of E1→E3 | `BORROW_OBJECTIVE`, `BASELINE` | CORE |
| L0067 | **Evolutionary music composition system with statistically modeled criteria** — Kowalczuk, Tatara, Bąk (2017) | Rule/statistical fitness; feature values modelled with Gaussian mean/variance | Simple interpretable distribution-based style objective | `BORROW_OBJECTIVE` | CORE |
| L0074 | **Groove2Groove: One-Shot Music Style Transfer with Supervision from Synthetic Data** — Cífka, Şimşekli, Richard (2020) | Explicit content-preservation and style-fit metrics; style profiles from symbolic events | Evaluation design is more valuable here than full model reproduction | `BORROW_METRIC`, `RELATED_WORK` | CORE FOR EVALUATION |
| L0093 | **MIDI-VAE: Modeling Dynamics and Instrumentation of Music with Applications to Style Transfer** — Brunner et al. (2018) | VAE/shared latent space plus style classifier | Classical neural/latent-style reference | `BASELINE`, `RELATED_WORK` | OPTIONAL |
| L0098 | **MuseMorphose: Full-Song and Fine-Grained Piano Music Style Transfer with One Transformer VAE** — Wu & Yang | Transformer-VAE with bar-level attribute control, including rhythmic intensity and polyphony | Reference for controllability, long context and attribute definitions | `ADAPT_METHOD`, `RELATED_WORK` | STRONG ADDITION / OPTIONAL MODEL |
| L0103 | **Musical Style Modification as an Optimization Problem** — Zalkow, Brand, Graf (2016) | Existing piece + local edits + multi-objective improvement; target/counterexample corpus; classifier and Markov objectives | Closest classical formulation to intended controlled transformation | `ADAPT_METHOD`, `BORROW_OBJECTIVE` | CORE |
| L0150 | **Symbolic Music Genre Transfer with CycleGAN** — Brunner et al. (2018) | Unpaired symbolic domain transfer with cycle/fidelity mechanisms | Historical neural baseline and context for current GAN attempt | `RELATED_WORK`, `BASELINE` | STRONG CONTEXT, LOW NEW IMPLEMENTATION |
| L0185 | **Encoding Musical Style with Transformer Autoencoders** — Choi et al. (2020) | Global style encoding combined with temporally distributed melody/structure representation | Strong conceptual support for separating content and style | `ADAPT_METHOD`, `RELATED_WORK` | STRONG ADDITION |
| L0207 | **From Generality to Mastery: Composer-Style Symbolic Music Generation via Large-Scale Pre-training** — Yao (2025) | Broad pretraining + lightweight composer adaptation | Modern composer-conditioned generation reference and data-scarcity strategy | `RELATED_WORK`, possible `BASELINE` | OPTIONAL |
| L0208 | **METEOR: Melody-aware Texture-controllable Symbolic Music Re-Orchestration via Transformer VAE** — Le & Yang (2025) | Explicit melodic fidelity plus texture control at bar/track level | Very strong conceptual match for preserve-melody / transform-accompaniment | `ADAPT_METHOD`, `BORROW_METRIC` | CORE CONCEPT / OPTIONAL MODEL |
| L0209 | **Composer Vector: Style-steering Symbolic Music Generation in a Latent Space** — Jiang et al. (2026) | Inference-time latent composer steering with continuous strength; style fusion/suppression | Modern reference for continuous style intensity | `RELATED_WORK`, experimental `BASELINE` | OPTIONAL |
| L0230 | **Quantifying Musical Style: Ranking Symbolic Music Based on Similarity to a Style** — Ens & Pasquier (2019) | StyleRank: feature-based corpus-relative style similarity/ranking | Independent evaluator/objective candidate, especially because E3 objective/evaluator agreement is weak | `BORROW_METRIC`, `BORROW_OBJECTIVE` | CORE |

## Cross-paper synthesis

### Family A — interpretable corpus-relative style models
`L0027 + L0034 + L0067 + L0103 + L0230`

Recurring structure:

```text
target corpus
    ↓
explicit representation/model
    ↓
style score
    ↓
search / ranking / evaluation
```

This is the highest-priority implementation family because it fits the existing E1/E3 infrastructure and enables controlled ablations.

### Family B — explicit content/style separation
`L0074 + L0150 + L0185 + L0098 + L0208`

Recurring ideas:
- style movement and content fidelity are separate objectives;
- content needs an explicit representation/constraint;
- one global similarity number is insufficient;
- named controllable attributes make evaluation more defensible.

This family should reshape the evaluation/content contract before it reshapes the model.

### Family C — modern composer-conditioned / latent control
`L0207 + L0209` plus neural predecessors

Recurring ideas:
- composer style is data-scarce;
- broad pretraining or latent structure can reduce per-composer data requirements;
- style strength can be continuous instead of only categorical.

Important for related work and optional extension, but not a reason to displace the interpretable core.

## Highest-value concrete adaptations

### 1. Classifier probability objective — L0034
Test whether optimizing a train-only composer classifier aligns better with an independent evaluator than current E3 profile distance.

Guardrail:
do not use the exact same fitted classifier as both optimization target and sole evaluation evidence.

### 2. Gaussian feature-profile objective — L0067

For selected feature `f_i`:

```text
score_i(x) = exp(-(f_i(x)-μ_i)^2 / (2σ_i^2))
```

Estimate `μ_i` and `σ_i` from training data only and balance semantic feature families.

### 3. Target-vs-counterexample objective — L0103

Prefer a relative signal where useful:

```text
target_affinity(candidate) - source_or_counterexample_affinity(candidate)
```

This may be more informative than absolute distance to a target centroid.

### 4. Event/style-profile metrics — L0074

Adapt the principle rather than copying groove-specific bins:
- onset position × duration;
- onset position × pitch/register;
- interval/time-difference profiles;
- texture/chord-size distributions.

### 5. StyleRank-like independent metric — L0230

Use it first as an evaluator. Consider it as an optimization objective only after its behavior on held-out real works is understood.

### 6. Pattern/Markov model — L0027 + L0103

A compact sequential model can provide a qualitatively different definition of style from global feature statistics.

## Papers not to reimplement by default

Do not schedule full reproductions of MIDI-VAE, CycleGAN, MuseMorphose, Transformer Autoencoder, Generality-to-Mastery, METEOR or Composer Vector unless:

1. public code/checkpoints are usable;
2. the representation can be mapped without rebuilding the whole project;
3. the experiment answers a question not already covered by E1/E3;
4. it can share the same style/content evaluation framework.
