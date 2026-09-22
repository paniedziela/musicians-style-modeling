"""Experimental standalone E4.6 inference using the frozen checkpoint decoder.

No training, calibration, classifier fitting or outer-test workflow is invoked.
"""
from functools import lru_cache
from pathlib import Path
import threading
import time

import numpy as np
import torch

from ..e3.inference import digest, json_bytes, validate_input, write_new
from ..evaluation.content import content_metrics, semantic_midi_equal
from ..midi.inference_io import export_performance
from ..midi.parser import MidiParser
from ..midi.printer import MidiPrettyPrinter
from .dataset import COMPOSERS, PieceSegments
from .experiment import E4Config, SCHEMA_VERSION, _geometry, _infer_piece, _threshold_values
from .model import ConditionalGenerator
from .segmentation import encode_piece, quantization_errors, stitch_segments, PITCH_LOW, PITCH_HIGH

DEFAULT_CHECKPOINT = Path("experiments/e4_asap_v3/best.pt")
_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _load(path: str, size: int, modified_ns: int):
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = E4Config(None, None, None, None, None)
    if (state.get("schema_version") != SCHEMA_VERSION
            or state.get("geometry") != _geometry(config)
            or state.get("composer_to_index") != {name: i for i, name in enumerate(COMPOSERS)}):
        raise ValueError("Checkpoint nie jest zgodnym modelem E4.6 (geometria, architektura lub style).")
    thresholds = state.get("thresholds")
    if not isinstance(thresholds, dict) or not {"onset", "frame"} <= thresholds.keys():
        raise ValueError("Checkpoint nie zawiera zamrożonych progów onset/frame.")
    if any(isinstance(thresholds[name], dict) and "threshold" not in thresholds[name] for name in ("onset", "frame")):
        raise ValueError("Checkpoint nie zawiera wartości zamrożonych progów onset/frame.")
    if any(not np.isfinite(value) or not 0 < value < 1 for value in _threshold_values(thresholds)):
        raise ValueError("Niepoprawne progi dekodera E4.6.")
    with torch.random.fork_rng(devices=[]):
        model = ConditionalGenerator(conv_dim=config.conv_dim, residual_blocks=config.residual_blocks)
    model.load_state_dict(state["generator"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.parameters()):
        raise ValueError("Checkpoint zawiera niefinitywne wagi.")
    model.eval()
    return model, {"schema": SCHEMA_VERSION, "epoch": state["epoch"],
                   "checkpoint": path, "checkpoint_sha256": digest(Path(path).read_bytes()),
                   "geometry": state["geometry"], "thresholds": thresholds,
                   "training_seed": state.get("seed"), "experimental": True,
                   "quality_validation": "not established; local E4.6 experiment was NO-GO"}


def checkpoint_info(path: Path = DEFAULT_CHECKPOINT):
    path = Path(path).resolve()
    stamp = path.stat()
    with _LOCK:
        _, info = _load(str(path), stamp.st_size, stamp.st_mtime_ns)
    return dict(info)


def infer(input_path: Path, target: str, output_path: Path, *, checkpoint: Path = DEFAULT_CHECKPOINT,
          midi_policy: str = "preserve", progress=None) -> dict:
    input_path, output_path = Path(input_path), Path(output_path)
    report_path = output_path.with_suffix(".json")
    if target not in COMPOSERS:
        raise ValueError("E4.6 obsługuje style: " + ", ".join(COMPOSERS))
    if output_path.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError("Wyjście musi mieć rozszerzenie .mid lub .midi.")
    if output_path.exists() or report_path.exists() or output_path.resolve() == input_path.resolve():
        raise ValueError("Nie nadpisuję istniejącego MIDI ani raportu; wybierz nowe wyjście.")
    if input_path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Wymagany MIDI do 10 MB.")
    data = input_path.read_bytes()
    source, common_warnings = validate_input(data, midi_policy=midi_policy)
    # The E4 tensor has no channel axis: do not silently merge separate instruments.
    if len({note.channel for note in source.notes}) != 1:
        raise ValueError("E4.6 wymaga jednego kanału nutowego; tensor nie rozdziela instrumentów. Wybierz E3 dla wielu kanałów.")
    mapped = encode_piece(source)
    if mapped.segment_count > 256:
        raise ValueError("Limit demonstracyjnej inferencji E4.6: 256 segmentów (1024 takty).")
    if any(bar.length_ticks < 16 for bar in mapped.bars):
        raise ValueError("E4.6 wymaga co najmniej 16 ticków w każdym takcie/segmencie metrum.")
    if not any(segment.nonempty for segment in mapped.segments):
        raise ValueError("Brak nut o dodatniej długości w zakresie modelu E4.6: MIDI 24–107.")
    piece = PieceSegments(digest(data), "external", "unknown", "inference", mapped)
    started = time.perf_counter()
    checkpoint = Path(checkpoint).resolve()
    stamp = checkpoint.stat()
    with _LOCK:
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            model, info = _load(str(checkpoint), stamp.st_size, stamp.st_mtime_ns)
            with torch.inference_mode():
                output, diagnostics = _infer_piece(model, piece, COMPOSERS.index(target), torch.device("cpu"),
                                                   info["thresholds"], progress=progress)
        finally:
            torch.set_num_threads(previous_threads)
    encoded, performance = export_performance(MidiPrettyPrinter().to_bytes(output), data, midi_policy)
    parsed = MidiParser().parse_bytes(encoded)
    if not parsed.notes or not semantic_midi_equal(output, parsed):
        raise ValueError("Wynik E4.6 nie przeszedł kontroli ponownego odczytu MIDI.")
    unchanged = semantic_midi_equal(source, parsed)
    quantized = stitch_segments(mapped)
    representation_only = not unchanged and semantic_midi_equal(quantized, parsed)
    status = "normalized_only" if unchanged and performance["removed"] else "unchanged" if unchanged else "transformed"
    warnings = list(common_warnings)
    warnings[:0] = [
        "E4.6 is experimental and its current validation result is NO-GO. A generated MIDI file is not evidence "
        "of successful style transfer.",
        "The model quantizes each bar to 16 steps in four-bar windows and may alter melody, rhythm, note count, "
        "and polyphony.",
        "The model tensor does not represent velocity or channel. The decoder copies them from the nearest note "
        "of the same pitch; newly introduced pitches use channel 0 and velocity 64.",
        "MIDI notes outside pitches 24–107 are copied. Zero-length notes within the model range are omitted, and "
        "overlapping notes of the same pitch may be merged.",
        "Key-signature metadata is copied from the source and may not match the generated notes.",
        "No style classification or independent-set evaluation was performed; inference does not run the outer test."]
    if unchanged:
        warnings.append("The notes are unchanged because the model/decoder returned an identity result or fallback.")
    elif representation_only:
        warnings.append("The output matches the E4 grid reconstruction of the input: this is a representation-only "
                        "change, with no demonstrated model-driven change.")
    errors = quantization_errors(mapped)
    report = {"schema": "e4-inference-1", "method": "E4.6", "experimental": True,
              "status": status, "note_material_unchanged": unchanged,
              "change_origin": "identity" if unchanged else "representation_only" if representation_only else "model_and_representation",
              "target_composer": target, "input_path": str(input_path.resolve()), "input_sha256": digest(data),
              "output_path": str(output_path.resolve()), "output_sha256": digest(encoded),
              "model": info, "device": "cpu", "threads": 2, "midi_processing": performance,
              "segments": mapped.segment_count, "diagnostics": diagnostics, "roundtrip_valid": True,
              "content": content_metrics(source, parsed), "representation_content": content_metrics(source, quantized),
              "quantization_max_onset_error_ticks": max((abs(row[1]) for row in errors), default=0),
              "elapsed_seconds": time.perf_counter()-started, "warnings": warnings}
    write_new(output_path, encoded)
    write_new(report_path, json_bytes(report))
    return report
