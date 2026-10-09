"""Exploratory equal-weight style-embedding mean, using frozen upstream weights.

Only short, single-program piano excerpts are supported. This is an inference
heuristic, not a trained multi-work composer model. Greedy decoding isolates
conditioning changes from sampling variation.
"""

import argparse
import json
from pathlib import Path

from confugue import Configuration
from groove2groove.io import NoteSequencePipeline
from groove2groove.models.roll2seq_style_transfer import Experiment
from museflow.note_sequence_utils import normalize_tempo
from note_seq import midi_io
import numpy as np

from run import WORKSPACE


def piano_sequence(path):
    sequence = midi_io.midi_file_to_note_sequence(str(path))
    if not sequence.notes or any(n.program != 0 or n.is_drum for n in sequence.notes):
        raise ValueError("Expected nonempty, piano-only MIDI: " + str(path))
    normalized = normalize_tempo(sequence, 60)
    if normalized.total_time > 32.01:
        raise ValueError("Select at most 32 quarter notes: " + str(path))
    return sequence, normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("content", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--style", type=Path, action="append", required=True)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    source, normalized_source = piano_sequence(args.content)
    references = [piano_sequence(path) for path in args.style]
    tempos = [seq.tempos[0].qpm for seq, _ in references]
    if not np.allclose(tempos, tempos[0]):
        raise ValueError("References must have a common tempo")
    model_dir = args.workspace / "checkpoints/v01"
    with (model_dir / "model.yaml").open("rb") as stream:
        config = Configuration.from_yaml(stream)
    experiment = config.configure(Experiment, logdir=str(model_dir), train_mode=False,
                                  sampling_seed=42)
    session = experiment.trainer.session
    experiment.trainer.load_variables(checkpoint_name="latest")
    tensors = experiment.model._inputs
    vectors = []
    for _, normalized in references:
        tokens = experiment.output_encoding.encode(normalized)
        vector = session.run(tensors["style_embedding"],
                             feed_dict={tensors["style_input"]: [tokens]})
        vectors.append(vector)
    mean_vector = np.mean(vectors, axis=0)
    content = experiment.input_encoding.encode(normalized_source)
    output_ids = session.run(experiment.model.greedy_outputs[1], feed_dict={
        tensors["content_input"]: [content],
        tensors["style_embedding"]: mean_vector,
        tensors["softmax_temperature"]: 0.6,
    })
    generated = experiment.output_encoding.decode(output_ids[0])
    velocity = int(np.mean([np.mean([n.velocity for n in seq.notes])
                            for seq, _ in references]) + 0.5)
    for note in generated.notes:
        note.velocity = velocity
        note.program = 0
    pipeline = NoteSequencePipeline(source, references[0][0], warp=True)
    list(pipeline.load())
    output = pipeline.postprocess([generated])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    midi_io.note_sequence_to_midi_file(output, str(args.output))
    args.output.with_suffix(".json").write_text(json.dumps({
        "content": str(args.content), "references": [str(p) for p in args.style],
        "conditioning": "equal arithmetic mean of frozen style embeddings",
        "decoding": "greedy", "embedding_shape": list(mean_vector.shape),
        "embedding_norms": [float(np.linalg.norm(v)) for v in vectors],
        "mean_embedding_norm": float(np.linalg.norm(mean_vector)),
        "output_notes": len(output.notes),
    }, indent=2) + "\n")
    session.close()
    print("Generated", args.output)


if __name__ == "__main__":
    main()
