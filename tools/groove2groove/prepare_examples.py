"""Make short, tempo-controlled ASAP score excerpts for the upstream baseline."""

import argparse
import io
import json
from pathlib import Path

import mido
import pretty_midi


PIECES = {
    "bach": "Bach/Prelude/bwv_846/midi_score.mid",
    "chopin": "Chopin/Etudes_op_10/1/midi_score.mid",
    "mozart": "Mozart/Piano_Sonatas/8-1/midi_score.mid",
}


def excerpt(source, destination, start_beat=0, beats=32, tempo=120):
    """Crop in quarter-note coordinates, retaining pitches and note velocities.

    Controllers are intentionally omitted: these are score-texture comparisons.
    Notes crossing the crop boundaries are clipped, not moved to another beat.
    """
    if destination.exists():
        raise FileExistsError(destination)
    if start_beat < 0 or beats <= 0 or tempo <= 0:
        raise ValueError("Use a nonnegative start and positive duration/tempo")
    # Some ASAP files put tempo/meter events outside track 0. Merge in memory
    # before PrettyMIDI reads them, so it sees all tempo changes correctly.
    raw = mido.MidiFile(str(source))
    merged = mido.MidiFile(type=0, ticks_per_beat=raw.ticks_per_beat)
    merged.tracks.append(mido.merge_tracks(raw.tracks))
    buffer = io.BytesIO()
    merged.save(file=buffer)
    buffer.seek(0)
    midi = pretty_midi.PrettyMIDI(buffer)
    output = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=480)
    output.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))
    piano = pretty_midi.Instrument(program=0, name=source.parent.name)
    end_beat = start_beat + beats
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            start = midi.time_to_tick(note.start) / midi.resolution
            end = midi.time_to_tick(note.end) / midi.resolution
            if start < end_beat and end > start_beat:
                piano.notes.append(pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=(max(start, start_beat) - start_beat) * 60 / tempo,
                    end=(min(end, end_beat) - start_beat) * 60 / tempo,
                ))
    if not piano.notes:
        raise ValueError("The selected excerpt has no notes")
    output.instruments.append(piano)
    output.write(str(destination))
    return len(piano.notes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asap", type=Path, default=Path("datasets/asap-dataset-1.2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for name, relative_path in PIECES.items():
        source = args.asap / relative_path
        destination = args.output / (name + ".mid")
        count = excerpt(source, destination)
        records.append({"name": name, "source": str(source), "file": destination.name,
                        "start_quarter": 0, "quarters": 32, "tempo": 120,
                        "notes": count})
    (args.output / "sources.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
