# STATUS — Research V2 entry point

> Draft generated from the current public `main` branch and the selected thesis literature.
> Baseline repository: `paniedziela/musicians-style-modeling`
> Inspected `main` commit: `594e70190708808516b93731c397d42e55b34371`

## Purpose

This file is the single operational entry point for the repository. It should answer what is finished, what is frozen, what is active, and which documents are authoritative.

## Current experimental state

### E1 — composer-style discrimination
**Status: CLOSED / GO**

- leakage-safe grouped evaluation;
- frozen 93-feature `composition_full` contract;
- balanced accuracy approximately `0.864`;
- useful as evidence that the representation contains composer-discriminative signal, not as proof of a universal definition of style.

### E2 — legacy global GA
**Status: CLOSED / FROZEN BASELINE**

Preserve it as a paired baseline. Do not improve it in place.

### E3 — constrained local optimization
**Status: CLOSED / GO WITH LIMITATIONS**

Reported:
- mean `Δp_target = 0.0377`;
- clustered 95% CI `[0.0285, 0.0460]`;
- paired E3−E2 improvement `0.0100`, CI `[0.0013, 0.0224]`.

Primary limitation:
- Pearson correlation between E3's internal style objective and the independent `Δp_target` evaluator is only `0.044`.

This is the main motivation for Research V2: compare alternative definitions of style and objectives before replacing the optimizer.

### E4.6 — conditional GAN inspired by StarGAN
**Status: CLOSED / NO-GO FOR THE CURRENT VARIANT**

Validation failed the frozen gate; outer test remained closed. Preserve the result and do not continue an open-ended E4.7/E4.8 tuning sequence.

## Research V2 goals

1. Compare symbolic MIR feature representations.
2. Compare several defensible definitions of target style.
3. Improve agreement between transformation objective and independent evaluation.
4. Keep content preservation independent from style gain.
5. Turn the selected literature into adaptations, metrics and baselines rather than 13 separate reimplementations.
6. Make the repository easier to navigate.

## Active workstreams

- **V2-A — MIR features:** custom93, musif, jSymbolic, selected music21, combinations, optional MusicXML/Partitura.
- **V2-B — style metrics/objectives:** classifier probability, Gaussian profiles, target-vs-counterexample, event profiles, StyleRank-like similarity, optional Markov/pattern model.
- **V2-C — controlled transfer:** initially reuse E3 as the stable transformation harness.
- **V2-D — neural comparison:** optional; only after the evaluation framework is stronger.

## Authoritative documents

Read in this order:

1. `docs/STATUS.md`
2. `docs/research/RESEARCH_CONTEXT.md`
3. `docs/research/PAPER_IMPLEMENTATION_MATRIX.md`
4. `docs/research/EXPERIMENT_REGISTRY.md`
5. `docs/research/ROADMAP.md`
6. frozen E1–E4 plans and `docs/results/*.md`

Older audits remain historical evidence, not the current source of truth.

## Repository rules

- Never overwrite completed experiment directories.
- Never reinterpret an existing schema name after changing semantics.
- Never alter E1b's 93-feature contract in place.
- Do not use E4 outer test for tuning.
- Any style profile/model must be train-only for the relevant fold.
- New feature backends get new versioned identifiers.
- Research variants belong in configs/experiment IDs, not permanent method branches.
- Extract shared abstractions only when at least two real experiments need them.

## Local assets

Keep datasets, results and PDFs outside Git and share them between worktrees through:

```text
MSM_DATA_ROOT
MSM_RESULTS_ROOT
MSM_LITERATURE_ROOT
```

Suggested external layout:

```text
<external-root>/
├── datasets/
├── results/
└── literature/
    ├── selected/
    └── old/
```

## Next decision gate

Before a new neural model:
1. merge the small Research V2 foundation;
2. run MIR representation comparison;
3. implement at least two independent style metrics/objectives;
4. test whether objective improvement agrees with independent evaluation;
5. only then decide whether a neural experiment adds unique evidence.
