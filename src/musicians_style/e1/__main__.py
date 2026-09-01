"""Command-line entry point for the staged E1 experiment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .asap import load_e1_config, write_e1_artifacts
from .classification import write_e1a_results, write_e1b_results
from .composition_features import (
    COMPOSITION_FEATURES_FILENAME,
    write_composition_feature_cache,
)
from .features import FEATURES_FILENAME, write_legacy_feature_cache
from .open_set import (
    OPEN_FEATURES_FILENAME,
    OPEN_MANIFEST_FILENAME,
    write_e1_open_results,
    write_open_data,
)
from .reporting import write_e1_closure_report
from .splits import SPLITS_FILENAME, write_e1_splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an isolated stage of experiment E1")
    parser.add_argument("--config", default="configs/e1_asap.yaml")
    parser.add_argument(
        "--stage",
        choices=(
            "audit", "splits", "features", "classify", "all",
            "e1b-features", "e1b-classify", "e1b",
            "open-data", "open-classify", "open", "report",
        ),
        default="audit",
        help=(
            "audit=E1.0, splits=E1.1, features/classify=E1.2, "
            "e1b-*=E1.3, open-*=E1-open, report=E1.4"
        ),
    )
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument(
        "--run-dir",
        help="Explicit new output directory for E1a/E1b classify; must be absent or empty.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--permutations",
        type=int,
        default=999,
        help="Fast group-level association permutations on fixed OOF predictions.",
    )
    parser.add_argument(
        "--retraining-permutations",
        type=int,
        default=0,
        help="Costly group-label permutations with model refitting (recommend 99 for a final run).",
    )
    parser.add_argument(
        "--skip-group-sensitivity",
        action="store_true",
        help="Skip the additional retraining with one sample selected per group.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Do not print per-fit progress (progress.jsonl is still written).",
    )
    parser.add_argument(
        "--importance-repeats",
        type=int,
        default=10,
        help="Held-out permutation-importance repeats per E1b fold and feature.",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated classifier names; defaults to all predeclared models.",
    )
    parser.add_argument(
        "--variants",
        help="Comma-separated feature variants; defaults to every cache variant.",
    )
    parser.add_argument(
        "--skip-open-rotations",
        action="store_true",
        help="Skip leave-one-known-composer-out controls in E1-open.",
    )
    parser.add_argument("--e1b-run-dir", help="Completed E1b run used by E1.4.")
    parser.add_argument("--open-run-dir", help="Optional completed E1-open run used by E1.4.")
    parser.add_argument(
        "--report-dir",
        help="E1.4 artifact directory; defaults to <e1b-run-dir>/report.",
    )
    parser.add_argument(
        "--report-doc",
        default="docs/results/E1.md",
        help="Tracked Markdown summary written by E1.4.",
    )
    args = parser.parse_args(argv)
    selected_models = (
        tuple(value.strip() for value in args.models.split(",") if value.strip())
        if args.models
        else None
    )
    selected_variants = (
        tuple(value.strip() for value in args.variants.split(",") if value.strip())
        if args.variants
        else None
    )
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
        if args.stage in {"e1b-features", "e1b"}:
            composition_features_path = write_composition_feature_cache(
                manifest_path, dataset_root=config.dataset_root
            )
            print(f"E1b feature cache: {composition_features_path}")
        else:
            composition_features_path = config.output_dir / COMPOSITION_FEATURES_FILENAME
        if args.stage in {"classify", "all"}:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
            output_dir = (
                Path(args.run_dir)
                if args.run_dir
                else Path(args.experiments_dir) / f"{config.experiment_name}_{timestamp}"
            )
            if output_dir.exists() and any(output_dir.iterdir()):
                raise ValueError(
                    f"run directory is not empty: {output_dir}; choose a new --run-dir"
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            progress_path = output_dir / "progress.jsonl"

            def report_progress(event: dict[str, object]) -> None:
                with progress_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                if args.quiet_progress:
                    return
                if event["event"] == "permutation_completed":
                    position = int(event["position"])
                    total = int(event["total_permutations"])
                    if position == 1 or position == total or position % 10 == 0:
                        print(
                            f"[perm {position:>3}/{total}] {event['variant']} | "
                            f"{event['model']} | null BA={event['null_balanced_accuracy']:.3f}",
                            flush=True,
                        )
                    return
                if event["event"] not in {"fit_started", "fit_completed"}:
                    return
                prefix = f"[{event['position']:>3}/{event['total_fits']}]"
                identity = (
                    f"{event['analysis']} | {event['variant']} | {event['model']} | "
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
                retraining_permutations=args.retraining_permutations,
                include_group_sensitivity=not args.skip_group_sensitivity,
                progress_callback=report_progress,
                model_names=selected_models,
                variant_names=selected_variants,
                config_path=args.config,
                manifest_path=manifest_path,
            )
            print(f"E1a results: {results_path}")
            print(f"E1a predictions: {predictions_path}")
        if args.stage in {"e1b-classify", "e1b"}:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
            output_dir = (
                Path(args.run_dir)
                if args.run_dir
                else Path(args.experiments_dir) / f"{config.experiment_name}_e1b_{timestamp}"
            )
            if output_dir.exists() and any(output_dir.iterdir()):
                raise ValueError(
                    f"run directory is not empty: {output_dir}; choose a new --run-dir"
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            progress_path = output_dir / "progress.jsonl"

            def report_e1b_progress(event: dict[str, object]) -> None:
                with progress_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                if args.quiet_progress:
                    return
                if event["event"] == "permutation_completed":
                    position = int(event["position"])
                    total = int(event["total_permutations"])
                    if position == 1 or position == total or position % 10 == 0:
                        print(
                            f"[perm {position:>3}/{total}] {event['variant']} | "
                            f"{event['model']} | null BA={event['null_balanced_accuracy']:.3f}",
                            flush=True,
                        )
                    return
                if event["event"] not in {"fit_started", "fit_completed"}:
                    return
                prefix = f"[{event['position']:>4}/{event['total_fits']}]"
                identity = (
                    f"{event['analysis']} | {event['variant']} | {event['model']} | "
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

            results_path, predictions_path, _ = write_e1b_results(
                composition_features_path,
                splits_path,
                output_dir,
                bootstrap_samples=args.bootstrap_samples,
                permutations=args.permutations,
                retraining_permutations=args.retraining_permutations,
                include_group_sensitivity=not args.skip_group_sensitivity,
                permutation_importance_repeats=args.importance_repeats,
                progress_callback=report_e1b_progress,
                model_names=selected_models,
                variant_names=selected_variants,
                config_path=args.config,
                manifest_path=manifest_path,
            )
            print(f"E1b results: {results_path}")
            print(f"E1b predictions: {predictions_path}")
        if args.stage in {"open-data", "open"}:
            (
                open_manifest_path,
                open_report_path,
                open_features_path,
                open_report,
            ) = write_open_data(config)
            print(f"E1-open manifest: {open_manifest_path}")
            print(f"E1-open quality report: {open_report_path}")
            print(f"E1-open feature cache: {open_features_path}")
            if not open_report["quality_gate"]["passed"]:
                return 2
        else:
            open_manifest_path = config.output_dir / OPEN_MANIFEST_FILENAME
            open_features_path = config.output_dir / OPEN_FEATURES_FILENAME
        if args.stage in {"open-classify", "open"}:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
            output_dir = (
                Path(args.run_dir)
                if args.run_dir
                else Path(args.experiments_dir) / f"{config.experiment_name}_open_{timestamp}"
            )
            if output_dir.exists() and any(output_dir.iterdir()):
                raise ValueError(
                    f"run directory is not empty: {output_dir}; choose a new --run-dir"
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            progress_path = output_dir / "progress.jsonl"

            def report_open_progress(event: dict[str, object]) -> None:
                with progress_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                if not args.quiet_progress and event["event"] == "open_fit_completed":
                    print(
                        f"[open {event['position']}/{event['total_fits']}] "
                        f"{event['scenario']} | {event['model']} | fold {event['fold']}",
                        flush=True,
                    )

            results_path, predictions_path, _ = write_e1_open_results(
                open_features_path,
                splits_path,
                output_dir,
                config=config,
                config_path=args.config,
                manifest_path=open_manifest_path,
                progress_callback=report_open_progress,
                include_rotations=not args.skip_open_rotations,
            )
            print(f"E1-open results: {results_path}")
            print(f"E1-open predictions: {predictions_path}")
        if args.stage == "report":
            if not args.e1b_run_dir:
                raise ValueError("--e1b-run-dir is required for --stage report")
            report_dir = Path(args.report_dir) if args.report_dir else Path(args.e1b_run_dir) / "report"
            report_path, closure = write_e1_closure_report(
                args.e1b_run_dir,
                report_dir,
                open_run_dir=args.open_run_dir,
                report_doc=args.report_doc,
            )
            print(f"E1.4 report: {report_path}")
            print(f"Decision: {closure['decision']}")
    except Exception as exc:  # CLI boundary: preserve a useful non-zero result
        print(f"E1 stage failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
