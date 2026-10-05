"""Run A+B verification, or explicitly reviewed later scientific stages."""
import argparse
import importlib.util
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['feasibility','evaluate-a','pilot-b','synthetic'])
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--distribution',type=Path)
    parser.add_argument('--java',type=Path)
    parser.add_argument('--reviewed-protocol',type=Path)
    parser.add_argument('--v205',type=Path)
    parser.add_argument('--external-bundle',type=Path)
    for name in ('data','results','literature'): parser.add_argument('--'+name+'-root')
    args=parser.parse_args(); checkout=Path(__file__).resolve().parents[1]
    if args.output.exists(): parser.error('output already exists')
    if args.mode in ('evaluate-a','pilot-b') and not args.reviewed_protocol:
        parser.error('scientific stage requires a reviewed protocol; implementation task stops before this')
    if args.mode!='synthetic' and (not args.distribution or not args.java): parser.error('distribution and java paths required')
    if args.mode in ('evaluate-a','pilot-b') and not args.v205: parser.error('V2-05 directory required')
    if args.mode=='pilot-b' and not args.external_bundle: parser.error('frozen Track A external bundle required')
    spec=importlib.util.spec_from_file_location('checkout_provenance_probe',checkout/'src/musicians_style/provenance.py')
    probe=importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
    provenance=probe.collect_provenance(checkout)
    if not provenance['import']['passed']: parser.error('checkout source import guard failed')
    from musicians_style import research_ab
    configuration={n:getattr(args,n+'_root') for n in ('data','results','literature')}
    if args.mode=='synthetic': result=research_ab.synthetic_verification(checkout,args.output,configuration)
    else:
        from musicians_style.jsymbolic_runtime import JSymbolicRuntime
        runtime=JSymbolicRuntime(checkout,args.distribution,args.java)
        if args.mode=='feasibility': result=research_ab.feasibility(checkout,args.output,runtime,configuration)
        elif args.mode=='evaluate-a': result=research_ab.evaluate_a(checkout,args.output,runtime,args.v205,args.reviewed_protocol,configuration)
        else: result=research_ab.pilot_b(checkout,args.output,runtime,args.v205,args.external_bundle,args.reviewed_protocol,configuration)
    print(f"Completed {args.mode}; report artifacts: {args.output.resolve()}; passed={result.get('passed', 'see summary')}")
    return 0 if result.get('passed',True) else 1

if __name__=='__main__': raise SystemExit(main())
