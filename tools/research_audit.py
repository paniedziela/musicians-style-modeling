"""Audit entry point: detect stale ambient imports before loading research code."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new audit directory; existing destinations are rejected")
    parser.add_argument("--scope", choices=("inventory", "baseline"), default="inventory")
    for name in ("data", "results", "literature"):
        parser.add_argument("--" + name + "-root")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output already exists; choose a fresh directory")
    checkout = Path(__file__).resolve().parents[1]
    # Load only the stdlib provenance probe by exact source path. This does not
    # modify sys.path or replace/repair the package which Python actually imports.
    spec = importlib.util.spec_from_file_location("checkout_provenance_probe", checkout / "src" / "musicians_style" / "provenance.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    provenance = probe.collect_provenance(checkout)
    if not provenance["import"]["passed"]:
        args.output.mkdir(parents=True, exist_ok=False)
        probe.write_json(args.output / "provenance.json", provenance)
        probe.write_json(args.output / "audit.json", {"passed": False, "scope": args.scope, "error": "checkout import guard failed"})
        message = "Import guard failed: " + str(provenance["import"]["error"])
        (args.output / "report.md").write_text(message + "\nUse an editable installation or set PYTHONPATH to this checkout's src; retry with a fresh output directory.\n", encoding="utf-8")
        print(message)
        return 2
    from musicians_style.research_audit import run_audit

    result = run_audit(checkout, args.output, scope=args.scope, configuration={name: getattr(args, name + "_root") for name in ("data", "results", "literature")}, provenance=provenance)
    print(f"Audit passed={result['passed']}; report: {args.output.resolve() / 'report.md'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
