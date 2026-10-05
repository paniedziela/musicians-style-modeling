# Track A reviewed execution protocol - 2026-10-05

The user explicitly authorized reviewing/committing/pushing the A+B implementation
and then running **Track A only**. This supersedes the implementation-only stop
boundary for A. B remains unauthorized and its approval file remains false.
No main change, merge, rebase, classifier/feature tuning or outer-test evaluation.

## Frozen inputs and scope

Use r00/f00/inner00 from the existing manifest/splits. Preflight metadata:
76 inner-training samples / 45 works; 44 inner-validation samples / 24 works.
All 30 r00/f00 outer-test originals are forbidden. Input metadata hashes and the
unchanged backend lock are checked against `experiments/research_ab_review_a.json`.
Only train+validation original MIDIs and the 18 completed, hash-matched V2-05
output MIDIs are extracted. No full-corpus feature cache or old fitted model is
loaded. Other folds' identity metadata are checked for leakage without opening
any original test MIDI or feature file.

jSymbolic remains the pinned 2.2 user distribution, exact 42-descriptor /
638-dimensional subset/configuration/schema and Java options from the implementation.
No descriptor is added, removed or altered. Git attributes preserve exact pinned
configuration/schema bytes across Windows/Linux checkouts; original SHA256 values
remain unchanged. Java 1.8.0_291 is recorded. No GPU/environment/package changes.

## Training and denominators

Extract both representations for the same declared train and validation samples.
Train both classifiers on the intersection of successful training extractions;
retain every excluded training identity/failure. The external pipeline is
TRAIN median imputation (all-missing columns -> zero), VarianceThreshold(0),
StandardScaler, existing V2-04 LogisticRegression configuration. custom93 uses
its existing VarianceThreshold(0), StandardScaler and identical classifier
settings. No validation/output fitting, calibration or model selection.

Persist learned parameters and serialized fits before validation/output scoring;
verify exact reload score equality. Save all failed validation/output rows and
report both available and common matched denominators. Failed rows are undefined,
not zero-affinity substitutes. Retained dimensions are the training variance
support, not a representation choice based on validation accuracy.

## Predeclared reporting and interpretation

Primary discrimination: equal-work, macro-composer fractional top-probability
correct credit, matching V2-04 rank/tie rules (1e-12); compare chance 1/3. Report
existing conditional work bootstrap (2,000 draws, seed1729) descriptively.
Piece-level balanced accuracy/macro-F1 and per-composer recalls can be secondary
summaries of the same saved predictions; they do not select features/models.

For the same 18 V2-05 outputs, report target-affinity movement, source drop and
margin separately; pooled and original objective/direction subsets, positive /
negative / null (1e-12) rates, matched sign agreement, Pearson/Spearman and
work aggregation. The 18 rows come from three source works; do not treat them
as 18 independent works. Probability scales are not calibrated across backends.

Assess whether jSymbolic provides complete/reliable coverage, nondegenerate
attribution above chance with uncertainty, and informative response on the saved
outputs. Disagreement with optimized custom93 is evidence to retain, not a reason
to tune the external evaluator. If useful, recommend freezing its exact artifact
as an external attribution proxy for B; this is not musical/perceptual validation
or authorization to execute B. No outcome-based numeric continuation threshold
or automatic B promotion is introduced.

## Execution audit

An independent Python file-open audit wrapper restricts dataset reads to the
manifest/splits and the declared 120 original paths, blocks test/cache reads and
all dataset writes, and records unique accesses plus blocked attempts. Java only
receives per-attempt neutral `input.mid` copies; its pinned execution path never
receives corpus directories/original paths. Existing V2-05 outputs are hash
validated by the runner. Hash those authorized inputs and pins before/after;
no outer-test originals are opened even for preservation hashing.

After logical commits and a clean-tree push, mark only the A review artifact
approved using this explicit user authorization, record the actual clean HEAD /
protocol/source/input hashes, and invoke the existing `evaluate-a` workflow once
into `experiments/research_ab_a_evaluation_reviewed`. Preserve failures/artifacts.
Stop for review after A. B protocol, features, model settings and objective remain
untouched. No B optimization or scientific pilot is run.
