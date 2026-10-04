"""V2-02 nine-training-sample feasibility and explicit custom93 equivalence audit."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh artifact directory")
    parser.add_argument("--musif-python", type=Path, required=True, help="isolated musif environment interpreter")
    parser.add_argument("--timeout", type=float, default=180, help="seconds per musif attempt")
    for name in ("data", "results", "literature"):
        parser.add_argument("--" + name + "-root")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    checkout = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("checkout_provenance_probe", checkout / "src/musicians_style/provenance.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    provenance = probe.collect_provenance(checkout)
    if not provenance["import"]["passed"]:
        args.output.mkdir(parents=True, exist_ok=False)
        probe.write_json(args.output / "provenance.json", provenance)
        probe.write_json(args.output / "audit.json", {"passed": False, "error": "checkout import guard failed"})
        print("Import guard failed; use an editable installation or checkout-first PYTHONPATH and a fresh directory.")
        return 2
    from musicians_style.feature_feasibility import run_feasibility
    result = run_feasibility(checkout, args.output, args.musif_python, timeout=args.timeout,
                             configuration={name: getattr(args, name + "_root") for name in ("data", "results", "literature")}, provenance=provenance)
    print(f"Audit passed={result['passed']}; report: {args.output.resolve() / 'report.md'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
