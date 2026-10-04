# ROADMAP — Research V2

This roadmap favors fast scientific iteration while keeping the repository understandable.

## Phase 0 — freeze and orient

### R0.1 Tag the current evidence

```bash
git tag research-v1-frozen
```

Do not move old experiment outputs.

### R0.2 Add navigation documents

Add:
- `docs/STATUS.md`;
- `docs/research/RESEARCH_CONTEXT.md`;
- `docs/research/PAPER_IMPLEMENTATION_MATRIX.md`;
- `docs/research/EXPERIMENT_REGISTRY.md`;
- `docs/research/ROADMAP.md`.

### R0.3 Externalize local roots

Introduce backwards-compatible resolution for:

```text
MSM_DATA_ROOT
MSM_RESULTS_ROOT
MSM_LITERATURE_ROOT
```

**Gate:** existing tests and old configs still work.

## Phase 1 — small Research V2 foundation

Goal: create only the abstractions required by E1c/E1d.

### R1.1 Feature backend contract

Introduce a narrow interface, for example:

```python
class FeatureBackend(Protocol):
    name: str
    version: str

    def extract_file(self, path: Path) -> Mapping[str, float]:
        ...
```

Do not move E1's implementation immediately. Wrap it.

### R1.2 Feature schema

Record:
- canonical feature name;
- backend;
- backend version;
- semantic family;
- units where meaningful;
- source format;
- missing-value policy.

### R1.3 Style metric contract

Introduce an explicitly fold-scoped fit/score contract:

```python
class StyleMetric(Protocol):
    def fit(self, train_samples, target_label, ...): ...
    def score(self, sample): ...
```

### R1.4 Content metric bundle

Do not invent one new content scalar. Return a structured bundle of existing metrics.

**Gate**
- custom93 reproduces the frozen E1 values exactly;
- identity behavior is unchanged;
- E2/E3/E4 results are untouched.

## Phase 2 — E1c MIR comparison

### R2.1 musif adapter
Validate first on a small canonical subset:
- deterministic columns;
- finite/missing-value behavior;
- stable sample mapping;
- runtime;
- format compatibility.

Then cache the full corpus.

### R2.2 jSymbolic adapter
Keep Java external.
Provide:
- documented invocation;
- exported feature file;
- parser to common schema;
- version metadata.

### R2.3 selected music21
Use where it adds clearly useful theoretical descriptors. Do not inflate dimensionality solely for feature count.

### R2.4 combined representations
Combine by schema, not anonymous array concatenation.

### R2.5 E1c run
Reuse grouped E1 protocol.

**Gate:** complete comparison report exists regardless of which backend wins.

## Phase 3 — E1d style-metric comparison

Implement evaluators before modifying E3.

### R3.1 Current E3 metric as named baseline
Expose the current style profile / RMS-z metric without changing frozen E3.

### R3.2 Classifier metric
Cross-fitted or held-out composer affinity.

### R3.3 Gaussian profile metric
L0067-inspired train-only target distributions.

### R3.4 Event-profile metric
Adapt L0074's principle to piano/composer style.

### R3.5 StyleRank-like metric
Implement or wrap the ranking concept if feasible.

### R3.6 Comparison study
Score:
- real held-out works;
- identity/source;
- E2 outputs;
- E3 outputs.

Measure:
- agreement;
- target ranking;
- direction effects;
- correlation with E3 internal objective.

**Gate:** at least two independent style measures are demonstrably usable.

## Phase 4 — E3 objective variants

Do not redesign operators first.

### R4.1 Objective injection
Refactor E3 only enough for a configurable style objective.

### R4.2 E3a classifier objective
Use separation from the final evaluator.

### R4.3 E3b Gaussian objective
Use train-only target distributions.

### R4.4 E3c relative/sequential objective
Only after E3a/b:
- target-vs-counterexample;
- Markov/pattern score.

### R4.5 Equal-budget comparison
Use the same inputs, constraints and evaluation budget.

**Gate:** select the most defensible objective based on independent style gain, content trade-off, stability and interpretability.

## Phase 5 — score-aware extension

Optional but strong:
use ASAP MusicXML through musif/Partitura.

Question:
does richer notation improve style modelling beyond score MIDI?

Do not let this block the core E1c/E1d/E3x result.

## Phase 6 — neural decision

Start from the literature matrix, not from the existing GAN code.

For every candidate ask:
- public code/checkpoint?
- representation compatible?
- transformation rather than only generation?
- explicit melody/content control?
- full-piece evaluation possible?
- implementation cost bounded?
- same evaluation harness usable?

Default:
keep E4.6 as the GAN result.
If a neural comparison is pursued, prefer adapting an existing controllable pretrained/autoencoding model over training another GAN from scratch.

## Priorities

### MUST
- authoritative `STATUS.md`;
- external local roots;
- versioned feature schemas;
- feature backend API;
- style metric API;
- E1c;
- E1d with at least two independent metrics;
- configurable E3 objective without altering frozen E3.

### SHOULD
- musif;
- jSymbolic;
- Gaussian profile objective;
- classifier objective;
- StyleRank-like evaluator;
- maintained paper matrix.

### COULD
- Partitura/MusicXML study;
- Markov/LZ/PST objective;
- OpenMusic/LZ external baseline;
- listening-study helper refinements.

### DEFER
- another tuned GAN variant;
- a large Transformer trained from scratch;
- arbitrary-composer few-shot transfer;
- performance-style modelling;
- total MIDI representation rewrite.

## Git/worktree strategy

### Step 1 — foundation worktree

```bash
git worktree add ../msm-v2 -b refactor/research-v2 main
```

Complete Phase 0–1 and merge it before branching experiments.

### Step 2 — at most two primary parallel worktrees

```bash
git worktree add ../msm-mir -b experiment/mir-features main
git worktree add ../msm-objectives -b experiment/style-objectives main
```

Create a neural worktree only after a Phase 6 GO.

### Worktree rule

Datasets, results and PDFs live outside each checkout and are resolved through environment/config roots.

## Suggested task granularity for a coding agent

Good task:

```text
Implement MIR-01 from ROADMAP.md:
add a FeatureBackend adapter around the frozen E1b extractor,
without moving or changing composition_features.py.
Add regression tests proving all 93 values and names are identical.
Do not modify E2–E4.
```

Bad task:

```text
Implement Research V2.
```

Every coding task should specify:
- research reason;
- files to read;
- files allowed to change;
- frozen contracts;
- acceptance tests;
- expected artifacts;
- explicit non-goals.
