"""Standalone E3 inference. No experiment runners or classifiers are loaded."""
from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import mido
import numpy as np

from ..evaluation.content import content_metrics, semantic_midi_equal
from ..midi.parser import MidiParser
from ..midi.printer import MidiPrettyPrinter
from ..midi.types import InternalRepr
from ..midi.inference_io import CORE_EVENTS, performance_events, performance_warnings, export_performance
from .algorithm import E3GeneticAlgorithm, SearchConfig
from .profile import TargetProfile, build_target_profile, style_vector
from .structure import analyse_structure

DEFAULT_PROFILES = Path("inference_workspace/profiles")
SCHEMA = "e3-inference-1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def load_profile(path: Path) -> tuple[TargetProfile, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Niekompatybilna wersja profilu: {path}")
    values = dict(payload["profile"])
    for name in ("feature_names", "feature_groups", "train_sample_ids", "train_group_ids", "train_sha256"):
        values[name] = tuple(values[name])
    sizes = {"mean": 67, "std": 67, "onset_histogram": 16, "duration_histogram": 6,
             "pitch_class_histogram": 12, "interval_histogram": 25, "chord_size_histogram": 8}
    for name, size in sizes.items():
        vector = np.asarray(values[name], dtype=float)
        if vector.shape != (size,) or not np.isfinite(vector).all() or (vector < 0).any():
            raise ValueError(f"Niepoprawne dane profilu: {name}")
        if name == "std" and (vector <= 0).any():
            raise ValueError("Profil wymaga dodatnich odchyleń standardowych.")
        if name.endswith("histogram") and not (np.isclose(vector.sum(), 1.0) or np.isclose(vector.sum(), 0.0)):
            raise ValueError(f"Histogram profilu nie jest znormalizowany: {name}")
        values[name] = vector
    profile = TargetProfile(**values)
    _, names, groups = style_vector(InternalRepr(480, (), (), 1))
    if profile.feature_names != names or profile.feature_groups != groups or profile.max_polyphony < 1:
        raise ValueError("Niekompatybilny schemat cech profilu.")
    if not profile.train_sha256 or not profile.train_sample_ids or not profile.train_group_ids:
        raise ValueError("Brak pochodzenia profilu.")
    expected = digest(json.dumps({"sample_ids": profile.train_sample_ids,
                                 "group_ids": profile.train_group_ids,
                                 "sha256": profile.train_sha256}, sort_keys=True).encode())
    if expected != profile.fingerprint:
        raise ValueError("Niezgodny fingerprint profilu.")
    if not isinstance(payload.get("provenance"), dict) or not payload["provenance"]:
        raise ValueError("Brak opisu pochodzenia profilu.")
    return profile, payload


def available_profiles(directory: Path = DEFAULT_PROFILES) -> dict[str, Path]:
    result = {}
    for path in sorted(Path(directory).glob("*.json")):
        profile, _ = load_profile(path)
        if profile.composer in result:
            raise ValueError(f"Powielony profil: {profile.composer}")
        result[profile.composer] = path
    return result


def prepare_profiles(*, manifest: Path, splits: Path, dataset_root: Path,
                     output: Path = DEFAULT_PROFILES, repeat: int = 0, fold: int = 0) -> list[str]:
    """Explicit one-time export of existing E3 train-only profiles, without fitting E1."""
    source = json.loads(manifest.read_text(encoding="utf-8"))
    split = json.loads(splits.read_text(encoding="utf-8"))
    repetition = next(item for item in split["repetitions"] if item["repeat"] == repeat)
    partition = next(item for item in repetition["folds"] if item["fold"] == fold)
    samples = {row["sample_id"]: row for row in source["samples"] if row["validation_status"] == "accepted"}
    rows = [samples[key] for key in partition["train"]["sample_ids"]]
    forbidden = [samples[key] for key in partition["test"]["sample_ids"]]
    composers = sorted({row["composer"] for row in rows})
    paths = [output / f"{index}.json" for index in range(len(composers))]
    if any(path.exists() for path in paths):
        raise ValueError("Profile już istnieją; wybierz nowy katalog --profiles-dir.")
    representations = {}
    for row in rows:
        path = (dataset_root / row["score_path"]).resolve()
        if not path.is_relative_to(dataset_root.resolve()):
            raise ValueError("Ścieżka profilu poza katalogiem danych.")
        if digest(path.read_bytes()) != row["sha256"]:
            raise ValueError(f"Niezgodny SHA danych treningowych: {path}")
        representations[row["sample_id"]] = MidiParser().parse(path)
    for composer, path in zip(composers, paths):
        profile = build_target_profile(composer, rows, representations, forbidden_rows=forbidden)
        values = {key: value.tolist() if isinstance(value, np.ndarray) else value
                  for key, value in asdict(profile).items()}
        payload = {"schema": SCHEMA, "profile": values, "provenance": {
            "manifest": str(manifest.resolve()), "manifest_sha256": digest(manifest.read_bytes()),
            "splits_sha256": digest(splits.read_bytes()), "repeat": repeat, "fold": fold,
            "scope": "outer-train only; standalone inference, not historical evaluation"}}
        write_new(path, json_bytes(payload))
        load_profile(path)
    return composers


def validate_input(data: bytes, *, midi_policy: str = "preserve") -> tuple[InternalRepr, list[str]]:
    if not data or len(data) > 10 * 1024 * 1024:
        raise ValueError("Wymagany MIDI do 10 MB.")
    try:
        raw = mido.MidiFile(file=io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"Niepoprawny plik MIDI: {exc}") from exc
    if raw.type not in (0, 1) or raw.ticks_per_beat <= 0:
        raise ValueError("Obsługiwane są SMF 0/1 z dodatnim PPQ; bez SMPTE i SMF 2.")
    _, performance = performance_events(raw, midi_policy)
    ignored = set()
    for track in raw.tracks:
        opened = {}
        for message in track:
            if message.time < 0:
                raise ValueError("Ujemny czas zdarzenia MIDI.")
            if message.is_meta and message.type not in CORE_EVENTS and message.type != "end_of_track":
                ignored.add(message.type)
            if message.type in {"note_on", "note_off"}:
                if message.channel == 9:
                    raise ValueError("Inferencja nie obsługuje perkusji na kanale 10.")
                key = (message.channel, message.note)
                on = message.type == "note_on" and message.velocity > 0
                opened[key] = opened.get(key, 0) + (1 if on else -1)
                if opened[key] < 0:
                    raise ValueError("Nuta note_off bez note_on.")
        if any(opened.values()):
            raise ValueError("Niezamknięta nuta; wymagane pary note_on/note_off w tej samej ścieżce.")
    piece = MidiParser().parse_bytes(data)
    if not 1 <= len(piece.notes) <= 10000:
        raise ValueError("Inferencja wymaga 1–10 000 nut (limit lokalnego interfejsu).")
    if raw.length > 900:
        raise ValueError("Limit inferencji wynosi 15 minut.")
    # Bound pathological meter grids before structure allocation.
    end = max(n.tick + n.duration_ticks for n in piece.notes)
    for event in piece.meta:
        if event.kind == "time_signature":
            num, den = event.payload["numerator"], event.payload["denominator"]
            ticks = piece.ticks_per_beat * num * 4
            if ticks % den or ticks // den < 1 or end / (ticks // den) > 10000:
                raise ValueError("Metrum niewyrażalne w tickach lub przekroczony limit 10 000 taktów.")
    if end / (piece.ticks_per_beat * 4) > 10000:
        raise ValueError("Przekroczony limit długości w taktach.")
    analyse_structure(piece)
    if not semantic_midi_equal(piece, MidiParser().parse_bytes(MidiPrettyPrinter().to_bytes(piece))):
        raise ValueError("Nuty nakładają się w sposób niezgodny z zapisem MIDI projektu (round-trip).")
    warnings = ["Tracks are merged on export; track names, trailing silence, and some metadata are not preserved."]
    if ignored:
        warnings.append("Ignored metadata: " + ", ".join(sorted(ignored)) + ".")
    if not any(event.kind == "time_signature" for event in piece.meta):
        warnings.append("No time signature was found; analysis assumes 4/4.")
    warnings.extend(performance_warnings(performance))
    return piece, warnings


def infer(input_path: Path, target: str, output_path: Path, *, profiles_dir: Path = DEFAULT_PROFILES,
          seed: int = 1729, config: SearchConfig = SearchConfig(), progress=None, midi_policy: str = "preserve") -> dict:
    input_path, output_path = Path(input_path), Path(output_path)
    report_path = output_path.with_suffix(".json")
    if output_path.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError("Wyjście musi mieć rozszerzenie .mid lub .midi.")
    if output_path.exists() or report_path.exists() or output_path.resolve() == input_path.resolve():
        raise ValueError("Nie nadpisuję istniejącego pliku MIDI ani raportu; wybierz nowe wyjście.")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed musi być w zakresie 0..2^32-1.")
    profiles = available_profiles(profiles_dir)
    if target not in profiles:
        raise ValueError(f"Brak profilu {target!r}. Dostępne: {', '.join(profiles) or 'brak; uruchom prepare-profiles'}")
    profile, package = load_profile(profiles[target])
    if input_path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Wymagany MIDI do 10 MB.")
    data = input_path.read_bytes()
    if digest(data) in profile.train_sha256:
        raise ValueError("Wejście należy do korpusu treningowego tego profilu; wybierz nowy utwór lub inny fold.")
    source, warnings = validate_input(data, midi_policy=midi_policy)
    warnings[:0] = ["E3 preserves the onset and duration of the highest note at each onset (Skyline), "
                    "but global transposition can also shift the melody.",
                    "E3 objective gain is neither a similarity percentage nor evidence of musical quality."]
    started = time.perf_counter()
    result = E3GeneticAlgorithm(config).run(source, profile, seed=seed, progress=progress)
    if not result.evaluation.constraints.feasible:
        raise ValueError("Brak dopuszczalnego wyniku, również identity nie spełnia ograniczeń E3.")
    output = result.output
    transpose = result.genome.transpose_semitones
    # Correct only exported key metadata, AFTER the unchanged historical search.
    if transpose and any(event.kind == "key_signature" for event in output.meta):
        major = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
        minor = ("Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm")
        pcs = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        def update(event):
            if event.kind != "key_signature":
                return event
            key = event.payload["key"]
            pc = pcs[key[0]] + key.count("#") - key.count("b")
            name = (minor if key.endswith("m") else major)[(pc + transpose) % 12]
            return replace(event, payload={"key": name})
        output = replace(output, meta=tuple(update(event) for event in output.meta))
        warnings.append("Exported key signatures were shifted by the selected transposition; E3 constraints were "
                        "evaluated before this metadata correction.")
    encoded, performance = export_performance(MidiPrettyPrinter().to_bytes(output), data, midi_policy)
    parsed = MidiParser().parse_bytes(encoded)
    if not semantic_midi_equal(output, parsed):
        raise ValueError("Wynik nie przeszedł kontroli ponownego odczytu MIDI.")
    unchanged = semantic_midi_equal(source, parsed)
    status = "normalized_only" if unchanged and performance["removed"] else "unchanged" if unchanged else "transformed"
    if unchanged:
        warnings.append("The output is unchanged because E3 found no better feasible transformation.")
    report = {"schema": SCHEMA, "method": "E3", "status": status, "target_composer": target,
              "note_material_unchanged": unchanged, "midi_processing": performance,
              "input_path": str(input_path.resolve()), "input_sha256": digest(data),
              "output_path": str(output_path.resolve()), "output_sha256": digest(encoded),
              "profile_fingerprint": profile.fingerprint, "profile_sha256": digest(profiles[target].read_bytes()),
              "profile_provenance": package["provenance"], "seed": seed, "config": asdict(config),
              "configuration_label": "default E3 search" if config == SearchConfig() else "custom/demo search",
              "elapsed_seconds": time.perf_counter() - started, "genome": asdict(result.genome),
              "transpose_semitones": transpose, "style_gain": result.evaluation.style_gain,
              "group_gains": result.evaluation.group_gains, "constraints_before_key_correction": asdict(result.evaluation.constraints),
              "content": content_metrics(source, parsed), "roundtrip_valid": True,
              "stop_reason": result.stop_reason, "history": result.history, "warnings": warnings}
    write_new(output_path, encoded)
    write_new(report_path, json_bytes(report))
    return report
