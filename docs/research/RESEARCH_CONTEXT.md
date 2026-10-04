# RESEARCH_CONTEXT — symbolic composer-style modelling and controlled transfer

## Research problem

The project studies symbolic musical style modelling and controlled composer-style transformation.

```text
source symbolic piece
        +
target composer style learned from a corpus
        ↓
controlled transformation
        ↓
output that moves toward the target style
while retaining explicitly protected source content
```

This is not primarily generation from scratch, one-shot imitation of a single target example, free-form continuation, or expressive performance-style transfer.

## Working definition of style

Style is treated operationally as a corpus-relative set of recurring symbolic characteristics.

Candidate dimensions:
- pitch/register and pitch-class usage;
- melodic intervals and contour;
- rhythm and metrical placement;
- harmony/chroma and transitions;
- texture/polyphony/chord-size behavior;
- repetition/pattern structure;
- ornamentation and accompaniment behavior.

Research V2 should explicitly compare several definitions instead of assuming one representation captures “true style”.

## Working definition of content

Content preservation remains independent from style similarity.

### PRESERVE
- piece-level temporal extent within a frozen tolerance;
- meter and essential MIDI structure;
- primary melodic identity according to the chosen melody representation;
- ordering of protected melodic events.

### SOFTLY PRESERVE
- melodic rhythm, depending on experiment;
- harmonic/chroma relation to source;
- phrase/section boundaries where reliable;
- note density and polyphony within explicit budgets.

### STYLE-TRANSFORMABLE
- accompaniment pitches and intervals;
- metrical placement and duration patterns within constraints;
- voicing/texture;
- bounded density/polyphony;
- later: ornamentation if explicitly introduced.

The partition must be experiment-specific and versioned.

## Current evidence

### E1
Shows that the current 93-feature representation contains substantial composer-discriminative information under grouped-by-work evaluation.

It does not prove universal composer attribution, perceptual style, or that classifier optimization produces convincing transfer.

### E2
Documents why unconstrained/global edits are a weak baseline for content-preserving transfer.

### E3
The strongest current transfer experiment because it uses explicit feasibility constraints.

The central unresolved issue is objective validity: its internal style objective correlates very weakly with the independent classifier-based evaluator.

### E4
E4.6 is a valid negative result for a compact conditional GAN. Indefinite tuning is not justified by the current evidence.

## Core Research V2 questions

### RQ1 — representation
Which symbolic feature representations reliably encode composer-discriminative information under leakage-safe grouped evaluation?

### RQ2 — style metric
Which corpus-relative style measures agree on real held-out works and provide useful evaluation of transformed outputs?

### RQ3 — optimization objective
Does replacing E3's current objective with literature-inspired alternatives improve agreement between optimization target and independent style evaluation?

### RQ4 — transfer trade-off
How much target-style movement can be achieved for a given content-preservation budget?

### RQ5 — optional neural comparison
Does one carefully selected neural method provide evidence beyond the interpretable optimization pipeline?

## Evaluation principles

### Two-axis evaluation
Report separately:
1. target-style movement;
2. source-content preservation.

Do not collapse them into one scalar too early.

### Independent style measures
Prefer at least two:
- held-out composer-classifier probability;
- train-only feature/profile distance;
- StyleRank-like corpus similarity;
- event/style-profile similarity;
- target-vs-counterexample score.

### Leakage prevention
Fit only on training data:
- normalization;
- profiles;
- classifiers;
- feature selection;
- thresholds;
- embeddings;
- target-vs-counterexample models.

### Grouping
Preserve grouping by musical work to avoid leakage of related material across evaluation partitions.

## Architecture direction

```text
MIDI / MusicXML
      ↓
representation / parsing adapters
      ↓
feature extractors
      ↓
style models / metrics
      ↓
transformation / search
      ↓
independent evaluation
      ↓
experiment orchestration + reports
```

Experiment packages E1–E4 remain reproducibility layers. Do not immediately move all code.

## MIR tooling policy

- **custom93:** frozen first-class baseline.
- **musif:** first high-priority external extractor.
- **jSymbolic:** independent external baseline; prefer CLI/export adapter.
- **music21:** selected theoretical descriptors and utilities; not automatically the canonical mutable representation.
- **Partitura:** score-aware/MusicXML experiment and possible future score-performance work.
- **MusPy:** optional metrics/preprocessing, not foundational.
- **MidiTok:** only for a concrete sequence/neural experiment.
- **OpenMusic + LZ:** external historical/pattern baseline; document reproducibly rather than tightly integrating into Python.

## Literature policy

The 13 selected papers are not 13 mandatory implementations.

Each is classified as one or more of:
- `REPRODUCE`;
- `ADAPT_METHOD`;
- `BORROW_OBJECTIVE`;
- `BORROW_METRIC`;
- `BASELINE`;
- `RELATED_WORK_ONLY`.

Repeated ideas across independent papers receive higher priority.

## Repository usability goals

Prefer:
- small modules with explicit responsibilities;
- config-driven variants;
- one current `STATUS.md`;
- paper-to-code mapping;
- explicit schema/version IDs;
- external data/result roots;
- preserved historical experiment packages.

Avoid:
- giant experiment files accumulating all variants;
- long-lived method branches;
- silently changing old config meanings;
- putting datasets, PDFs or checkpoints into Git.

## Non-goals

Research V2 does not require:
- universal arbitrary-composer transfer;
- adding a new composer without retraining;
- full score voice separation;
- expressive performance modelling;
- production deployment;
- training a large symbolic foundation model from scratch;
- implementing every neural paper;
- proving one definitive mathematical definition of musical style.
