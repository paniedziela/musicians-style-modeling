import argparse
from pathlib import Path
import sys

from .dataset import COMPOSERS
from .inference import DEFAULT_CHECKPOINT, checkpoint_info, infer


def main(argv):
    parser = argparse.ArgumentParser(prog="python -m musicians_style.e4", description="Eksperymentalna inferencja E4.6 (NO-GO)")
    commands = parser.add_subparsers(dest="command", required=True)
    styles = commands.add_parser("styles")
    run = commands.add_parser("infer")
    for command in (styles, run):
        command.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--target", choices=COMPOSERS, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--midi-policy", choices=("preserve", "score-only", "strict"), default="preserve")
    args = parser.parse_args(argv)
    try:
        if args.command == "styles":
            info = checkpoint_info(args.checkpoint)
            print("E4.6 eksperymentalne, epoka", info["epoch"], "—", ", ".join(COMPOSERS))
        else:
            report = infer(args.input, args.target, args.output, checkpoint=args.checkpoint,
                           midi_policy=args.midi_policy,
                           progress=lambda row: print(f"Segment {row['segment']}/{row['segments']}", flush=True))
            for warning in report["warnings"]:
                print("WARNING: " + warning)
            print(f"Status: {report['status']}; zmiana: {report['change_origin']}; MIDI: {args.output}")
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"E4.6: {exc}", file=sys.stderr)
        return 1
