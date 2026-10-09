"""Focused checks for preprocessing and interpretation, without loading TF."""

from pathlib import Path
import tempfile
import unittest

import mido
import pretty_midi

from inspect_results import compare
from prepare_examples import excerpt
from run import ROOT, run_pair


class BaselineTests(unittest.TestCase):
    def test_crop_uses_beats_with_tempo_changes_outside_track_zero(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT / "inference_workspace")) as directory:
            source = Path(directory) / "source.mid"
            target = Path(directory) / "target.mid"
            midi = mido.MidiFile(ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack())
            midi.tracks.append(mido.MidiTrack([
                mido.MetaMessage("set_tempo", tempo=1000000),
                mido.Message("note_on", note=60, velocity=80),
                mido.MetaMessage("set_tempo", tempo=250000, time=480),
                mido.Message("note_off", note=60, time=960),
                mido.Message("note_on", note=64, velocity=90),
                mido.Message("note_off", note=64, time=480),
            ]))
            midi.save(str(source))
            excerpt(source, target, start_beat=1, beats=1)
            notes = pretty_midi.PrettyMIDI(str(target)).instruments[0].notes
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0].pitch, 60)
            self.assertEqual(notes[0].velocity, 80)
            self.assertAlmostEqual(notes[0].start, 0)
            self.assertAlmostEqual(notes[0].end, 0.5)
            self.assertAlmostEqual(compare(notes, notes)["pitch_time_f1"], 1)
            with self.assertRaises(FileExistsError):
                excerpt(source, target)

    def test_harmony_metric_does_not_claim_octave_or_melody_preservation(self):
        source = [pretty_midi.Note(80, 60, 0, 16)]
        octave_up = [pretty_midi.Note(80, 72, 0, 16)]
        metrics = compare(source, octave_up)
        self.assertAlmostEqual(metrics["quarter_chroma_cosine"], 1)
        self.assertEqual(metrics["pitch_time_f1"], 0)
        self.assertEqual(metrics["top_pitch_agreement"], 0)

    def test_existing_result_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT / "inference_workspace")) as directory:
            target = Path(directory) / "existing.mid"
            target.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                run_pair("missing.mid", "missing.mid", target)
            self.assertEqual(target.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
