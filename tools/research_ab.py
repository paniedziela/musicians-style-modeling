"""Run jSymbolic feasibility or the explicitly reviewed external evaluation."""
import argparse
import importlib.util
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['feasibility','evaluate-a'])
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--distribution',type=Path)
    parser.add_argument('--java',type=Path)
    parser.add_argument('--reviewed-protocol',type=Path)
    parser.add_argument('--v205',type=Path)
    for name in ('data','results','literature'): parser.add_argument('--'+name+'-root')
    args=parser.parse_args(); checkout=Path(__file__).resolve().parents[1]
    if args.output.exists(): parser.error('output already exists')
    if args.mode=='evaluate-a' and not args.reviewed_protocol:
        parser.error('scientific stage requires a reviewed protocol; implementation task stops before this')
    if (not args.distribution or not args.java): parser.error('distribution and java paths required')
    if args.mode=='evaluate-a' and not args.v205: parser.error('V2-05 directory required')
    spec=importlib.util.spec_from_file_location('checkout_provenance_probe',checkout/'src/musicians_style/provenance.py')
    probe=importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
    provenance=probe.collect_provenance(checkout)
    if not provenance['import']['passed']: parser.error('checkout source import guard failed')
    from musicians_style import research_ab
    configuration={n:getattr(args,n+'_root') for n in ('data','results','literature')}
    from musicians_style.jsymbolic_runtime import JSymbolicRuntime
    runtime=JSymbolicRuntime(checkout,args.distribution,args.java)
    if args.mode=='feasibility': result=research_ab.feasibility(checkout,args.output,runtime,configuration)
    else: result=research_ab.evaluate_a(checkout,args.output,runtime,args.v205,args.reviewed_protocol,configuration)
    print(f"Completed {args.mode}; report artifacts: {args.output.resolve()}; passed={result.get('passed', 'see summary')}")
    return 0 if result.get('passed',True) else 1

if __name__=='__main__': raise SystemExit(main())
