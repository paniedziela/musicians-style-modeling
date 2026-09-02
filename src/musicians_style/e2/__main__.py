"""Command line entry point for the staged E2 experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .experiment import E2Experiment, load_e2_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m musicians_style.e2",
        description="Run the leakage-safe E2 current-GA baseline experiment.",
    )
    parser.add_argument("--config", required=True, help="E2 YAML configuration.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "pilot", "run", "report", "all"),
        help="Experiment stage to execute.",
    )
    parser.add_argument(
        "--run-dir",
        help="Artifact directory. Defaults to output_dir from the configuration.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: configuration, normally 1).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_e2_config(Path(args.config))
        experiment = E2Experiment(
            config,
            run_dir=Path(args.run_dir) if args.run_dir else None,
            workers=args.workers,
        )
        if args.stage == "prepare":
            experiment.prepare()
        elif args.stage == "pilot":
            experiment.ensure_prepared()
            experiment.run_stage("pilot")
        elif args.stage == "run":
            experiment.ensure_prepared()
            experiment.run_stage("full")
        elif args.stage == "report":
            experiment.write_report()
        else:
            experiment.prepare()
            experiment.run_stage("pilot")
            experiment.run_stage("full")
            experiment.write_report()
        return 0
    except KeyboardInterrupt:
        print("E2 interrupted; completed task records are safe to resume.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"E2 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI
    raise SystemExit(main())
