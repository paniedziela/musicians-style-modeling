"""Inspect controlled 32-quarter, 120 BPM examples; no composer-style claims."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi


def read_notes(path):
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = [note for instrument in midi.instruments for note in instrument.notes]
    if not notes:
        raise ValueError("Empty MIDI: " + str(path))
    if any(not (0 <= n.pitch <= 127 and 1 <= n.velocity <= 127
                and 0 <= n.start < n.end) for n in notes):
        raise ValueError("Invalid notes: " + str(path))
    return notes


def roll(notes, steps_per_quarter=12, quarters=32):
    result = np.zeros((128, steps_per_quarter * quarters), dtype=bool)
    for note in notes:
        start = int(round(note.start * 2 * steps_per_quarter))
        end = int(round(note.end * 2 * steps_per_quarter))
        result[note.pitch, max(0, start):min(result.shape[1], end)] = True
    return result


def compare(content, output):
    source_roll = roll(content)
    output_roll = roll(output)
    chroma_source = np.array([source_roll[p::12].sum(axis=0) for p in range(12)])
    chroma_output = np.array([output_roll[p::12].sum(axis=0) for p in range(12)])
    # Pool each quarter separately so harmonic progression matters, not only key.
    source_beats = chroma_source.reshape(12, 32, 12).sum(axis=2)
    output_beats = chroma_output.reshape(12, 32, 12).sum(axis=2)
    numerator = (source_beats * output_beats).sum(axis=0)
    denominator = np.linalg.norm(source_beats, axis=0) * np.linalg.norm(output_beats, axis=0)
    valid = denominator > 0
    cosine = np.divide(numerator, denominator, out=np.zeros(32), where=valid)
    joint = source_roll.any(axis=0) & output_roll.any(axis=0)
    source_top = 127 - source_roll[::-1].argmax(axis=0)
    output_top = 127 - output_roll[::-1].argmax(axis=0)
    intersection = (source_roll & output_roll).sum()
    return {
        "quarter_chroma_cosine": float(cosine.mean()),
        "pitch_time_f1": float(2 * intersection / (source_roll.sum() + output_roll.sum())),
        "top_pitch_agreement": float((source_top[joint] == output_top[joint]).mean())
        if joint.any() else 0.0,
        "active_quarters": int(output_beats.any(axis=0).sum()),
    }


def plot_pair(content, style, output, path):
    figure, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, sharey=True)
    for axis, notes, title in zip(axes, (content, style, output),
                                  ("Content", "Style reference", "Generated")):
        for note in notes:
            axis.plot([note.start * 2, note.end * 2], [note.pitch, note.pitch],
                      linewidth=1.7)
        axis.set_title(title, loc="left")
        axis.set_ylabel("MIDI pitch")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Quarter notes (all examples at 120 BPM)")
    axes[-1].set_xlim(0, 32)
    figure.tight_layout()
    figure.savefig(str(path), dpi=140)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = args.run / "inspection"
    report.mkdir(exist_ok=False)
    rows = []
    for path in sorted((args.run / "outputs").glob("*.mid")):
        content_name, rest = path.stem.split("_to_", 1)
        style_name = rest.split("_seed")[0]
        content = read_notes(args.run / "inputs" / (content_name + ".mid"))
        style = read_notes(args.run / "inputs" / (style_name + ".mid"))
        output = read_notes(path)
        row = {"file": path.name, "content_notes": len(content),
               "style_notes": len(style), "output_notes": len(output),
               "first_onset_quarter": min(n.start for n in output) * 2,
               "last_offset_quarter": max(n.end for n in output) * 2,
               "unique_pitches": len(set(n.pitch for n in output)),
               "mean_duration_quarters": np.mean([n.end - n.start for n in output]) * 2,
               "unique_velocities": len(set(n.velocity for n in output))}
        row.update(compare(content, output))
        row["style_quarter_chroma_cosine"] = compare(style, output)["quarter_chroma_cosine"]
        rows.append(row)
        plot_pair(content, style, output, report / (path.stem + ".png"))
    if not rows:
        raise ValueError("No outputs found")
    with (report / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print("{}: {} notes, content chroma {:.3f}, pitch-time F1 {:.3f}, top {:.3f}".format(
            row["file"], row["output_notes"], row["quarter_chroma_cosine"],
            row["pitch_time_f1"], row["top_pitch_agreement"]))


if __name__ == "__main__":
    main()
