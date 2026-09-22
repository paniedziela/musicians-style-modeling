"""CLI for the staged E3 experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m musicians_style.e3", epilog="Inferencja: infer --help; profile: styles / prepare-profiles --help")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("prepare", "pilot", "run", "report", "all"))
    parser.add_argument("--run-dir", help="Override output.run_dir for this invocation.")
    parser.add_argument("--workers", type=int, default=None, help="Override the YAML worker-process count for this invocation.")
    parser.add_argument("--e2-run-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"infer", "styles", "prepare-profiles"}:
        from .inference_cli import main as inference_main
        return inference_main(argv)
    from .experiment import E3Experiment, load_e3_config
    args = build_parser().parse_args(argv)
    try:
        experiment = E3Experiment(load_e3_config(args.config), run_dir=args.run_dir, workers=args.workers)
        if args.stage == "prepare":
            experiment.prepare()
        elif args.stage == "pilot":
            experiment.run_stage("pilot")
        elif args.stage == "run":
            experiment.run_stage("full")
        elif args.stage == "report":
            experiment.write_report(args.e2_run_dir)
        else:
            experiment.prepare(); experiment.run_stage("pilot"); experiment.run_stage("full"); experiment.write_report(args.e2_run_dir)
        return 0
    except KeyboardInterrupt:
        print("E3 interrupted; completed tasks can be resumed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"E3 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
