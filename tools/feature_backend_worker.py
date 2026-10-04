"""Single MIDI musif extraction in a dependency-isolated subprocess."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    environment = {"executable": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                   "python": sys.version, "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    try:
        if sys.prefix == sys.base_prefix:
            raise ValueError("musif worker requires an isolated virtual environment")
        from musicians_style.provenance import import_provenance
        guard = import_provenance(Path(__file__).resolve().parents[1])
        if not guard["passed"]:
            raise ValueError("worker checkout import guard failed")
        environment["import"] = guard
        from musicians_style.feature_backends.musif import extract
        result = {"status": "success", **extract(args.input)}
    except Exception as exc:
        result = {"status": "failure", "failure": {"type": type(exc).__name__, "message": str(exc)}}
    result.update(extraction_seconds=time.perf_counter() - started, environment=environment)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
