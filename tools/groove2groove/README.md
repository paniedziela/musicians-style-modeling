# Groove2Groove: independent pretrained baseline

This workflow runs the original frozen **v01** checkpoint on CPU. It does not
use or modify E1–E3/V2 code, datasets, checkpoints, or results. No training is
performed. Upstream model code is unmodified.

Source: [Cífka, Şimşekli and Richard (2020)](https://doi.org/10.1109/TASLP.2020.3019642),
*Groove2Groove: One-Shot Music Style Transfer with Supervision from Synthetic Data*.
The local paper is `Literatura/selected/L0074.pdf`.
[Official code](https://github.com/cifkao/groove2groove) and MuseFlow retain their
upstream BSD licenses in their downloaded source directories. The wrapper code
here calls their APIs; it does not redistribute the checkpoint or upstream code.

## Run

From the repository root, with network access for setup:

```bash
bash tools/groove2groove/setup.sh
export MPLCONFIGDIR="$PWD/inference_workspace/groove2groove/mpl"
export OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2
G2G_PY="$PWD/inference_workspace/groove2groove/env/bin/python"

# Use a NEW output directory; existing results are deliberately refused.
"$G2G_PY" tools/groove2groove/run_examples.py \
  --output experiments/groove2groove/my_comparison
"$G2G_PY" tools/groove2groove/inspect_results.py \
  experiments/groove2groove/my_comparison
"$G2G_PY" tools/groove2groove/test_baseline.py
```

One pair of already prepared MIDIs:

```bash
"$G2G_PY" tools/groove2groove/run.py content.mid style.mid output.mid --seed 42
```

The wrapper uses original `run-midi`, program-based instrument filtering,
eight-bar content segments, sampling at temperature 0.6, and batch size 1.
`--greedy` disables sampling. For arbitrary files, upstream converts inputs to
beat-relative time and outputs at the reference's first tempo. **Crop the style
reference yourself**: upstream uses the entire reference, not an eight-bar crop.
Use short references, ideally eight bars. The diagnostics specifically assume
32 quarters at 120 BPM and must not be applied unchanged to arbitrary MIDIs.

Setup uses workspace-local micromamba, Python 3.7.12, CPU TensorFlow 1.15.5,
NumPy 1.18.5 and pinned dependencies. Upstream originally used TF 1.12;
TF 1.15.5 restores this checkpoint successfully without source patches.
LMDB 1.4.1 supplies a wheel, avoiding an unavailable C compiler. Setup was
validated in this Linux environment, including a rerun on the installed runtime.
The environment, caches, source and 107 MB checkpoint occupy about 1.8 GB under
ignored `inference_workspace/groove2groove/`. No GPU or system installation is needed.

Pinned source revisions:

- Groove2Groove: `fe016926e62973a66f0b49d697374effe0f9e627`
- MuseFlow: `e5642fd48687a6d3165e213eaab79d5d78d4c0d5`

## Actual examples and observations

The completed run is `experiments/groove2groove/asap_01/`:

- `inputs/`: first eight bars of Bach BWV 846 Prelude, Chopin Op. 10 No. 1,
  Mozart K. 310 first movement, from ASAP **score** MIDIs.
- `outputs/`: all nine ordered pairs, including three self-reference controls;
  each MIDI has its command, timing, and upstream log alongside it.
- `inspection/`: nine content/reference/output piano-roll plots and `summary.csv`.

Preparation preserves pitches, score quarter-note positions and note velocities,
clips notes at excerpt boundaries, merges piano parts and removes controllers.
It sets every excerpt to 120 BPM and 4/4. Tempo/meter events outside track zero
are merged in memory before parsing. This is a score-texture experiment, not a
performance-expression experiment. All three main excerpts begin in 4/4.

All nine generated MIDIs parse successfully and contain 65–223 positive-duration
notes spanning 10–21 distinct pitches. Runs took approximately 15–20 seconds
each on this CPU setup. They are nonempty, varying textures, not copied inputs.
However, this does **not** establish satisfactory musical quality:

| Content → reference | Output notes | Content chroma cosine | Pitch-time F1 |
|---|---:|---:|---:|
| Bach → Mozart | 223 | 0.748 | 0.534 |
| Chopin → Bach | 95 | 0.360 | 0.038 |
| Chopin → Chopin | 65 | 0.874 | 0.370 |
| Mozart → Mozart | 193 | 0.528 | 0.534 |

Chroma cosine compares duration-weighted pitch classes within corresponding
quarter-note windows; silent windows score zero. Pitch-time F1 compares binary
pitch occupancy at 12 frames/quarter. These are lightweight diagnostics, **not
the paper's evaluation protocol**. Highest-active-pitch agreement ranges from
1.3% to 13.3% across the nine runs; it is only an upper-voice proxy, not melody
recognition. The tests explicitly show that perfect chroma similarity can coexist
with zero pitch agreement after octave displacement.
There is only one main excerpt per composer; key, register and texture are
confounded with composer identity. No composer-discrimination claim is tested.

Visual inspection shows that Bach → Mozart replaces the arpeggiation with a
chordal texture while retaining some harmonic material. Even Chopin → Chopin
loses the defining wide arpeggios, retaining mainly lower-register material.
Bach → Chopin begins with four silent quarters; several Mozart outputs also
contain substantial silence. Self-reference is not expected to be an identity
mapping, but these controls expose the practical loss of recognizable figuration.
Upstream logs report one or two decoder event inconsistencies in four of nine
runs; its normal decoder cleanup remains enabled. MIDI validity is assessed
after that decoding/postprocessing, not on raw generated events.

The v01 model does not generate velocity variation: upstream assigns the
reference's mean velocity to every output note. Its synthetic Band-in-a-Box
training domain is accompaniment, not complete classical piano works. The
paper itself recommends retaining the content melody separately (section VI-B)
and discusses accompaniment limitations (section VII). These outputs support
using it as a texture-rewriting baseline; they do not demonstrate composer-style
transfer with recognizable melodic content. No listening panel or subjective
listening assessment was performed; conclusions here use notes and plots.

`checks_01/repeat_seed42.mid` exactly matches the original Bach → Mozart notes.
Changing to seed 43 yields 190 notes, chroma cosine 0.587 and pitch-time F1 0.472.
Thus fixed-seed repeatability holds here, but individual musical outcomes vary.
Reproducibility across other TF builds/hardware is not established.

## Optional multi-reference exploration

`mean_style.py` encodes each reference separately into its frozen 500-dimensional
style vector and feeds their equal arithmetic mean to the original decoder.
It supports short piano-only inputs and uses greedy decoding. This is motivated
by the paper's style interpolation experiment (section VI-E), but **is not a
trained or validated composer representation**. References are not concatenated.

```bash
"$G2G_PY" tools/groove2groove/mean_style.py content.mid mean_output.mid \
  --style reference_work_1.mid --style reference_work_2.mid
```

Completed artifacts are in `experiments/groove2groove/mean_style_01/`:

- `single_chopin.mid`: Bach content, Chopin Op. 10 No. 1 reference.
- `single_chopin_op25.mid`: same content, Op. 25 No. 1 reference.
- `two_chopin_works_aligned.mid`: equal mean of those two references.
- `comparison.json`: note counts and content diagnostics for all three.
- `both_references.png`: content, both actual references, and the pooled output.
  The exploratory three-panel PNGs use Op. 10 No. 1 as a common comparison reference.

The Op. 25 No. 1 excerpt starts at quarter 1, after its one-quarter pickup,
and spans 32 quarters at 120 BPM. Its source is
`datasets/asap-dataset-1.2/Chopin/Etudes_op_25/1/midi_score.mid`;
`prepare_examples.excerpt(..., start_beat=1)` reproduces it. Use the
`inputs/chopin_op25_1_bar1.mid` reference. The earlier `two_chopin_works.mid`
used a pickup-inclusive excerpt and is retained only as an exploratory artifact.

The single-reference embedding path matches upstream greedy output exactly
(37 notes, pitches, timings and velocities). The other single-reference output
has 14 notes; their embedding mean produces 115 notes with content chroma cosine
0.587, compared with 0.096 and 0.442 for the endpoints. It still has zero
highest-active-pitch agreement with content. This demonstrates that the frozen
decoder can accept pooled conditioning and produce a distinct result, not that
pooling improves composer fidelity. Greedy and sampled results are not directly
comparable. More references, training and full-dataset evaluation were not run.

Three focused tests cover beat-accurate cropping with misplaced/changing tempo
metadata, the distinction between harmony and pitch preservation, and refusal
to overwrite results. Dependency checks and setup shell syntax checks pass.
Earlier setup probes are preserved in `asap_pilot_01/` and `asap_pilot_02/`;
use `asap_01/` for the complete comparison. All generated outputs remain ignored.
