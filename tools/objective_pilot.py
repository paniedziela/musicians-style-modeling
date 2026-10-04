"""Run only the fixed 18-search V2-05 pilot in a fresh directory."""
import argparse
import importlib.util
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    for name in ('data', 'results', 'literature'):
        parser.add_argument('--' + name + '-root')
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    if args.output.exists():
        parser.error('output already exists; choose a fresh directory')
    spec = importlib.util.spec_from_file_location('checkout_provenance_probe', checkout / 'src/musicians_style/provenance.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    provenance = probe.collect_provenance(checkout)
    if not provenance['import']['passed']:
        args.output.mkdir(parents=True, exist_ok=False)
        probe.write_json(args.output / 'provenance.json', provenance)
        probe.write_json(args.output / 'audit.json', {'passed': False, 'failures': ['checkout import guard failed']})
        return 2
    from musicians_style.objective_pilot import run_pilot
    result = run_pilot(checkout, args.output, provenance=provenance,
                       configuration={n:getattr(args,n + '_root') for n in ('data', 'results', 'literature')})
    print(f"Pilot passed={result['passed']}; runtime={result['elapsed_seconds']:.2f}s; report={args.output.resolve() / 'report.md'}")
    return 0 if result['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
