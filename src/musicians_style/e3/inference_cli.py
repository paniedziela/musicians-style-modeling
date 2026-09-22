"""CLI commands for standalone E3, separate from historical experiment flags."""
import argparse
from pathlib import Path
import sys

from .algorithm import SearchConfig
from .inference import DEFAULT_PROFILES, available_profiles, infer, prepare_profiles


def main(argv):
    parser = argparse.ArgumentParser(prog="python -m musicians_style.e3")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("styles", help="Lista zweryfikowanych profili")
    prepare = commands.add_parser("prepare-profiles", help="Jednorazowy eksport profili train, bez treningu modeli")
    run = commands.add_parser("infer", help="Transformacja własnego MIDI przez E3")
    for command in (listing, prepare, run):
        command.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES)
    prepare.add_argument("--manifest", type=Path, default=Path("datasets/derived/e1_asap/manifest.json"))
    prepare.add_argument("--splits", type=Path, default=Path("datasets/derived/e1_asap/splits.json"))
    prepare.add_argument("--dataset-root", type=Path, default=Path("datasets/asap-dataset-1.2"))
    prepare.add_argument("--repeat", type=int, default=0)
    prepare.add_argument("--fold", type=int, default=0)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--target", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seed", type=int, default=1729)
    run.add_argument("--generations", type=int, default=60)
    run.add_argument("--population-size", type=int, default=32)
    run.add_argument("--midi-policy", choices=("preserve", "score-only", "strict"), default="preserve",
                     help="preserve: zachowaj CC (w tym sustain CC64) i pitch bend; "
                          "score-only: jawnie usuń zdarzenia wykonawcze")
    args = parser.parse_args(argv)
    try:
        if args.command == "styles":
            profiles = available_profiles(args.profiles_dir)
            print("\n".join(profiles) or "Brak profili. Uruchom prepare-profiles.")
        elif args.command == "prepare-profiles":
            names = prepare_profiles(manifest=args.manifest, splits=args.splits, dataset_root=args.dataset_root,
                                     output=args.profiles_dir, repeat=args.repeat, fold=args.fold)
            print("Zapisano profile: " + ", ".join(names))
        else:
            config = SearchConfig(generations=args.generations, population_size=args.population_size)
            result = infer(args.input, args.target, args.output, profiles_dir=args.profiles_dir,
                           seed=args.seed, config=config, midi_policy=args.midi_policy,
                           progress=lambda row: print(f"Generacja {row['generation']}: zysk E3 {row['best_style_gain']:.6f}", flush=True))
            for warning in result["warnings"]:
                print("WARNING: " + warning)
            print(f"Status: {result['status']}; MIDI: {args.output}; raport: {args.output.with_suffix('.json')}")
        return 0
    except KeyboardInterrupt:
        print("Przerwano inferencję E3.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"E3: {exc}", file=sys.stderr)
        return 1
