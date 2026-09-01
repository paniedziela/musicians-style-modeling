"""Command-line entry point for the staged E1 experiment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .asap import load_e1_config, write_e1_artifacts
from .classification import write_e1a_results
from .features import FEATURES_FILENAME, write_legacy_feature_cache
from .splits import SPLITS_FILENAME, write_e1_splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an isolated stage of experiment E1")
    parser.add_argument("--config", default="configs/e1_asap.yaml")
    parser.add_argument(
        "--stage",
        choices=("audit", "splits", "features", "classify", "all"),
        default="audit",
        help="audit=E1.0, splits=E1.1, features/classify=E1.2",
    )
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Do not print per-fit progress (progress.jsonl is still written).",
    )
    args = parser.parse_args(argv)
    try:
        config = load_e1_config(args.config)
        manifest_path = config.output_dir / "manifest.json"
        if args.stage in {"audit", "all"}:
            manifest_path, report_path, report = write_e1_artifacts(config)
            print(f"Manifest: {manifest_path}")
            print(f"Quality report: {report_path}")
            print(
                f"Accepted: {report['accepted_count']}/{report['candidate_count']}; "
                f"gate: {'PASS' if report['quality_gate']['passed'] else 'FAIL'}"
            )
            if not report["quality_gate"]["passed"]:
                return 2
        if args.stage in {"splits", "all"}:
            splits_path = write_e1_splits(
                manifest_path,
                seeds=config.split_seeds,
                outer_splits=config.outer_splits,
                inner_splits=config.inner_splits,
            )
            print(f"Splits: {splits_path}")
        else:
            splits_path = config.output_dir / SPLITS_FILENAME
        if args.stage in {"features", "all"}:
            features_path = write_legacy_feature_cache(manifest_path)
            print(f"Feature cache: {features_path}")
        else:
            features_path = config.output_dir / FEATURES_FILENAME
        if args.stage in {"classify", "all"}:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            output_dir = Path(args.experiments_dir) / f"{config.experiment_name}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            progress_path = output_dir / "progress.jsonl"

            def report_progress(event: dict[str, object]) -> None:
                with progress_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                if args.quiet_progress or event["event"] not in {"fit_started", "fit_completed"}:
                    return
                prefix = f"[{event['position']:>3}/{event['total_fits']}]"
                identity = (
                    f"{event['variant']} | {event['model']} | "
                    f"repeat {event['repeat'] + 1}/{len(config.split_seeds)} | "
                    f"fold {event['fold'] + 1}/{config.outer_splits}"
                )
                if event["event"] == "fit_started":
                    print(f"{prefix} START {identity}", flush=True)
                else:
                    print(
                        f"{prefix} DONE  {identity} | "
                        f"{event['elapsed_seconds']:.1f}s | "
                        f"BA={event['balanced_accuracy']:.3f}",
                        flush=True,
                    )

            results_path, predictions_path, _ = write_e1a_results(
                features_path,
                splits_path,
                output_dir,
                bootstrap_samples=args.bootstrap_samples,
                permutations=args.permutations,
                progress_callback=report_progress,
            )
            print(f"E1a results: {results_path}")
            print(f"E1a predictions: {predictions_path}")
    except Exception as exc:  # CLI boundary: preserve a useful non-zero result
        print(f"E1 stage failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
