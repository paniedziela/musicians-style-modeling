"""Command-line entry point for the isolated E1.0 audit."""

from __future__ import annotations

import argparse
import sys

from .asap import load_e1_config, write_e1_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and audit the ASAP E1.0 manifest")
    parser.add_argument("--config", default="configs/e1_asap.yaml")
    args = parser.parse_args(argv)
    try:
        config = load_e1_config(args.config)
        manifest_path, report_path, report = write_e1_artifacts(config)
    except Exception as exc:  # CLI boundary: preserve a useful non-zero result
        print(f"E1.0 audit failed: {exc}", file=sys.stderr)
        return 1
    print(f"Manifest: {manifest_path}")
    print(f"Quality report: {report_path}")
    print(
        f"Accepted: {report['accepted_count']}/{report['candidate_count']}; "
        f"gate: {'PASS' if report['quality_gate']['passed'] else 'FAIL'}"
    )
    return 0 if report["quality_gate"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
