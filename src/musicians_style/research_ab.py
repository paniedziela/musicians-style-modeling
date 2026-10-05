"""Explicit A+B workflows. Feasibility uses only nine declared outer-training MIDIs."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .asset_paths import resolve_asset_roots
from .feature_feasibility import select_pilot, compare_attempts, attempt
from .feature_backends import custom93
from .jsymbolic_runtime import JSymbolicRuntime
from .external_classifier import fit_external, score_external
from .provenance import collect_provenance, sha256_file, write_json
from .style_metrics import assert_disjoint, LOGISTIC, rank_credit
from .style_audit import validate_splits, cluster_summary, agreement, movement


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def workflow_provenance(checkout):
    provenance=collect_provenance(checkout)
    provenance['source_sha256']={str(p.relative_to(checkout)):sha256_file(p) for p in sorted((checkout/'src/musicians_style').rglob('*.py'))}
    return provenance


def fresh(checkout,output,configuration=None):
    roots=resolve_asset_roots(checkout,configuration)
    results=Path(roots['results']['path']).resolve()
    output=output.resolve()
    if output.exists() or output==results or not output.is_relative_to(results):
        raise ValueError('fresh child of results root required')
    if any(output.is_relative_to(p.resolve()) for p in (results.iterdir() if results.exists() else []) if p.is_dir()):
        raise ValueError('cannot nest in existing experiment')
    output.mkdir(parents=True)
    return roots


def inputs(roots):
    data=Path(roots['data']['path'])
    manifest=data/'derived/e1_asap/manifest.json'
    splits=data/'derived/e1_asap/splits.json'
    return read(manifest),read(splits),data/'asap-dataset-1.2',dict(manifest=sha256_file(manifest),splits=sha256_file(splits))


def verified_source(root,row):
    path=(root/row['score_path']).resolve()
    if not path.is_relative_to(root.resolve()) or sha256_file(path)!=row['sha256']:
        raise ValueError('source path/hash mismatch: '+row['sample_id'])
    return path


def extraction(runtime,root,row,output):
    try: return runtime.extract(verified_source(root,row),output)
    except Exception as exc:
        record=dict(status='failure',failure=dict(type=type(exc).__name__,message=str(exc)))
        output.mkdir(parents=True,exist_ok=True); write_json(output/'result.json',record)
        return record


def feasibility(checkout,output,runtime,configuration=None):
    roots=fresh(checkout,output,configuration)
    manifest,splits,source_root,hashes=inputs(roots)
    cohort=select_pilot(manifest,splits)
    write_json(output/'protocol.json',dict(stage='feasibility',selection='V2-02 nine lexical r00/f00 OUTER TRAIN samples',
        repetitions=2,lock=runtime.lock,scientific_evaluation=False))
    write_json(output/'pilot_manifest.json',dict(cohort=cohort,input_metadata_hashes=hashes))
    write_json(output/'provenance.json',workflow_provenance(checkout))
    rows=[]
    for repeat in (0,1):
        for i,row in enumerate(cohort):
            result=extraction(runtime,source_root,row,output/f'r{repeat}_{i:02d}')
            rows.append(dict(sample_id=row['sample_id'],repetition=repeat,**result))
            write_json(output/'attempts.json',rows)
            print(f'jSymbolic {repeat+1}/2 {i+1}/9 {result["status"]}',flush=True)
    comparisons=[dict(sample_id=row['sample_id'],**compare_attempts(rows[i],rows[i+9])) for i,row in enumerate(cohort)]
    summary=dict(attempts=len(rows),successes=sum(r['status']=='success' for r in rows),failures=sum(r['status']=='failure' for r in rows),
        repeatability=all(c['passed'] for c in comparisons),comparisons=comparisons,
        feature_count=len(rows[0].get('schema',[])),extraction_seconds=sum(r.get('extraction_seconds',0) for r in rows),
        passed=len(rows)==18 and all(r['status']=='success' for r in rows) and all(c['passed'] for c in comparisons),
        opened_original_ids=[r['sample_id'] for r in cohort],outer_test_original_access=0)
    write_json(output/'summary.json',summary)
    write_json(output/'failures.json',[r for r in rows if r['status']=='failure'])
    return summary


def custom_attempt(root,row):
    try: return attempt(custom93.extract,verified_source(root,row))
    except Exception as exc:
        return dict(status='failure',failure=dict(type=type(exc).__name__,message=str(exc)))


def partition(manifest,splits,*,two_works=False):
    samples=[r for r in manifest['samples'] if r.get('validation_status')=='accepted']
    folds=validate_splits(samples,splits)
    outer=next(f for repeat,f in folds if repeat==0 and f['fold']==0)
    inner=next(f for f in outer['inner_folds'] if f['fold']==0)
    by_id={r['sample_id']:r for r in samples}
    train=[by_id[s] for s in sorted(inner['train']['sample_ids'])]
    validation=[by_id[s] for s in sorted(inner['validation']['sample_ids'])]
    forbidden=[r for r in samples if r['sample_id'] not in {r['sample_id'] for r in train}]
    assert_disjoint(train,forbidden)
    cohort=[]
    if two_works:
        for composer in sorted({r['composer'] for r in train}):
            candidates=[r for r in validation if r['composer']==composer]
            groups=sorted({r['group_id'] for r in candidates})[:2]
            if len(groups)!=2: raise ValueError('two validation works per composer required')
            cohort += [min((r for r in candidates if r['group_id']==group),key=lambda r:r['sample_id']) for group in groups]
    return train,validation,forbidden,cohort


def reviewed(path,stage,hashes,lock):
    if path is None: raise ValueError('scientific execution requires a reviewed protocol file')
    protocol=read(path)
    if protocol.get('approved_for_execution') is not True or protocol.get('stage')!=stage:
        raise ValueError('protocol is not approved for this stage')
    if protocol.get('input_metadata_hashes')!=hashes or protocol.get('backend_lock')!=lock:
        raise ValueError('reviewed input/backend fingerprints differ')
    if stage=='track_b' and any(not protocol.get(k) for k in ('continuation_thresholds','listening_design','external_bundle_sha256','objective_bundle_sha256')):
        raise ValueError('B review must predeclare continuation/listening and both frozen fits')
    return protocol


def _score(pipeline,vector):
    probabilities=pipeline.predict_proba(np.asarray(vector).reshape(1,-1))[0]
    return dict(zip(pipeline.classes_,map(float,probabilities)))


def evaluate_a(checkout,output,runtime,v205,review_path,configuration=None):
    # Review checked before original MIDI/features or existing output reads.
    roots=resolve_asset_roots(checkout,configuration)
    manifest,splits,source_root,hashes=inputs(roots)
    protocol=reviewed(review_path,'track_a',hashes,runtime.lock)
    train,validation,forbidden,_=partition(manifest,splits)
    fresh(checkout,output,configuration)
    write_json(output/'protocol.json',protocol)
    write_json(output/'cohort.json',dict(train=train,validation=validation,forbidden=forbidden))
    write_json(output/'provenance.json',workflow_provenance(checkout))
    ext={}; custom={}; all_rows=train+validation
    for i,row in enumerate(all_rows):
        sid=row['sample_id']
        ext[sid]=extraction(runtime,source_root,row,output/f'extraction_{i:03d}')
        custom[sid]=custom_attempt(source_root,row)
        write_json(output/'extractions.json',dict(external=ext,custom93=custom))
    matched=[r for r in train if ext[r['sample_id']]['status']==custom[r['sample_id']]['status']=='success']
    if {r['composer'] for r in matched}!={r['composer'] for r in train}:
        raise ValueError('matched training extraction lacks class coverage')
    ext_model,metadata=fit_external(matched,forbidden,ext)
    x=np.asarray([custom[r['sample_id']]['values'] for r in matched])
    if x.shape!=(len(matched),93) or not np.isfinite(x).all(): raise ValueError('invalid custom93 training matrix')
    custom_model=Pipeline([('variance',VarianceThreshold()),('scale',StandardScaler()),('model',LogisticRegression(**LOGISTIC))]).fit(x,[r['composer'] for r in matched])
    metadata.update(declared_train=train,matched_training_ids=[r['sample_id'] for r in matched],
        excluded_training_ids=[r['sample_id'] for r in train if r not in matched],
        custom93_fit=dict(variance_support=custom_model[0].get_support().tolist(),
            scaler_mean=custom_model[1].mean_.tolist(),scaler_scale=custom_model[1].scale_.tolist(),
            coef=custom_model[-1].coef_.tolist(),intercept=custom_model[-1].intercept_.tolist(),classes=custom_model.classes_.tolist()))
    bundle=dict(external=ext_model,custom93=custom_model,metadata=metadata,backend_lock=runtime.lock,
                input_metadata_hashes=hashes)
    joblib.dump(bundle,output/'frozen_external.joblib')
    write_json(output/'fit.json',metadata)
    bundle=joblib.load(output/'frozen_external.joblib')
    check_ids=[r['sample_id'] for r in matched]
    if score_external(ext_model,metadata['schema'],ext,check_ids)!=score_external(bundle['external'],metadata['schema'],ext,check_ids):
        raise ValueError('serialized external fit score mismatch')
    if not np.array_equal(custom_model.predict_proba(x),bundle['custom93'].predict_proba(x)):
        raise ValueError('serialized custom93 fit score mismatch')
    # Fit persistence precedes any validation prediction/output extraction.
    write_json(output/'fit_manifest.json',dict(bundle_sha256=sha256_file(output/'frozen_external.joblib'),
        fit_metadata_sha256=sha256_file(output/'fit.json'),persisted_before_scoring=True))
    predicted={r['sample_id']:r for r in score_external(bundle['external'],metadata['schema'],ext,[r['sample_id'] for r in validation])}
    real=[]; score_map={}
    for row in validation:
        sid=row['sample_id']; result=predicted[sid]
        scores=dict(external=result.get('affinities'),custom93=_score(bundle['custom93'],custom[sid]['values']) if custom[sid]['status']=='success' else None)
        score_map[sid]=scores
        record=dict(sample_id=sid,group_id=row['group_id'],composer=row['composer'],scores=scores,
            external_failure=result.get('failure'),custom93_failure=custom[sid].get('failure'))
        for backend,values in scores.items():
            record[backend+'_credit']=rank_credit(values,row['composer'])[0] if values else None
        record['matched']=all(values is not None for values in scores.values())
        real.append(record)
    previous=read(v205/'per_output.json')
    if len(previous)!=18 or len({r['task_id'] for r in previous})!=18 or any(r.get('status')!='completed' for r in previous): raise ValueError('exactly 18 completed V2-05 outputs required')
    outputs=[]
    for i,row in enumerate(previous):
        sid=row['source_id']; original=next((r for r in validation if r['sample_id']==sid),None)
        if original is None or row['source_sha256']!=original['sha256']: raise ValueError('V2-05 source outside declared validation')
        path=(v205/'tasks'/row['task_id']/'output.mid').resolve()
        if not path.is_relative_to(v205.resolve()) or sha256_file(path)!=row['output_sha256']: raise ValueError('V2-05 output path/hash mismatch')
        extraction_result=runtime.extract(path,output/f'v205_{i:02d}')
        external=score_external(bundle['external'],metadata['schema'],{'output':extraction_result},['output'])[0]
        own=attempt(custom93.extract,path)
        scores=dict(external=external.get('affinities'),custom93=_score(bundle['custom93'],own['values']) if own['status']=='success' else None)
        record=dict(task_id=row['task_id'],sample_id=sid,group_id=original['group_id'],composer=original['composer'],target=row['target'],scores=scores,
                    failures=dict(external=external.get('failure'),custom93=own.get('failure')))
        for backend in scores:
            before=score_map[sid][backend]; after=scores[backend]
            record[backend+'_movement']=movement(before,after,original['composer'],row['target']) if before and after else None
            record[backend+'_delta']=record[backend+'_movement']['delta'] if record[backend+'_movement'] else None
        record['matched']=all(record[b+'_delta'] is not None for b in scores)
        outputs.append(record)
        write_json(output/'output_scores.json',outputs)
    ranking={b:dict(all=cluster_summary(real,b+'_credit',stratified=True),
        matched=cluster_summary([r for r in real if r['matched']],b+'_credit',stratified=True)) for b in ('external','custom93')}
    movement_summary={b:dict(all=cluster_summary(outputs,b+'_delta'),matched=cluster_summary([r for r in outputs if r['matched']],b+'_delta')) for b in ('external','custom93')}
    summary=dict(real_ranking=ranking,output_movement=movement_summary,
        agreement=agreement(outputs,'external_delta','custom93_delta'),real_rows=len(real),outputs=len(outputs),
        external_bundle_sha256=sha256_file(output/'frozen_external.joblib'),
        limitations='descriptive conditional work bootstrap; attribution/movement is not musical validity; external evaluator must never be optimized')
    write_json(output/'real_scores.json',real); write_json(output/'summary.json',summary)
    return summary


def pilot_b(checkout,output,runtime,v205,external_path,review_path,configuration=None):
    from .accompaniment_search import PitchSource,Logistic93Objective,local_search
    from .content_metrics import observe_midi,measure_content
    from .objective_pilot import PilotObjective,run_search,content_failures
    from .e3.algorithm import SearchConfig
    from .midi.printer import MidiPrettyPrinter
    roots=resolve_asset_roots(checkout,configuration)
    manifest,splits,source_root,hashes=inputs(roots)
    protocol=reviewed(review_path,'track_b',hashes,runtime.lock)
    # Verify the frozen fits BEFORE loading original validation sources or searching.
    objective_path=v205/'fits/inner00.joblib'
    if sha256_file(external_path)!=protocol['external_bundle_sha256'] or sha256_file(objective_path)!=protocol['objective_bundle_sha256']:
        raise ValueError('reviewed frozen fit hash mismatch')
    train,validation,forbidden,cohort=partition(manifest,splits,two_works=True)
    objective_metadata=read(v205/'fits/inner00.json')
    fields=('sample_id','group_id','sha256','composer')
    identities=lambda rows:sorted(tuple(r[k] for k in fields) for r in rows)
    if identities(objective_metadata['train'])!=identities(train) or identities(objective_metadata['forbidden'])!=identities(forbidden):
        raise ValueError('objective fit partition mismatch')
    external=joblib.load(external_path)
    if external['backend_lock']!=runtime.lock or external['input_metadata_hashes']!=hashes:
        raise ValueError('external fit backend/input mismatch')
    ext_metadata=external['metadata']
    if identities(ext_metadata['declared_train'])!=identities(train) or identities(ext_metadata['forbidden'])!=identities(forbidden):
        raise ValueError('external fit partition mismatch')
    frozen=joblib.load(objective_path)['style']
    fresh(checkout,output,configuration)
    write_json(output/'protocol.json',protocol)
    write_json(output/'provenance.json',workflow_provenance(checkout))
    composers=sorted({r['composer'] for r in cohort})
    tasks=[dict(source=row,target=target,seed=seed,method=method) for row in cohort
           for target in composers if target!=row['composer'] for seed in (1729,1730) for method in ('e3_zero','pitch_search')]
    write_json(output/'pilot_manifest.json',dict(cohort=cohort,tasks=tasks,proposals=512,
        external_bundle_sha256=sha256_file(external_path),objective_bundle_sha256=sha256_file(objective_path),
        budget_note='E3 initial population includes identity: 32+16*30=512 proposals; 545 population/identity requests including cache. Pitch search 512 proposals/requests including identity. Mechanism comparison, not optimizer ablation.'))
    source_bytes={r['sample_id']:verified_source(source_root,r).read_bytes() for r in cohort}
    results=[]; identities_out=[]
    for row in cohort:
        sid=row['sample_id']; observation=observe_midi(source_bytes[sid])
        ext_attempt=runtime.extract(verified_source(source_root,row),output/f'identity_{sid}')
        affinities=score_external(external['external'],ext_metadata['schema'],{sid:ext_attempt},[sid])[0]
        identities_out.append(dict(sample_id=sid,external=affinities))
    source_scores={r['sample_id']:r['external'].get('affinities') for r in identities_out}
    write_json(output/'identity_scores.json',identities_out)
    write_json(output/'optimization_started.json',dict(fits_frozen=True,stage='development pilot',scientific_success_claim=False))
    for index,task in enumerate(tasks):
        row=task['source']; sid=row['sample_id']; target=task['target']; seed=task['seed']; method=task['method']
        directory=output/f'task_{index:02d}'; directory.mkdir()
        record=dict(source_id=sid,group_id=row['group_id'],composer=row['composer'],target=target,seed=seed,method=method,status='failed')
        started=time.perf_counter()
        try:
            original=source_bytes[sid]; observation=observe_midi(original)
            objective=Logistic93Objective(frozen['logistic'],target)
            baseline=objective(original)
            if method=='pitch_search':
                source=PitchSource.from_bytes(original)
                result=local_search(source,objective,seed=seed,proposals=512,changed_fraction=.20)
                data=result.midi; history=result.history; accounting=result.accounting; gain=result.gain
                write_json(directory/'notes.json',source.note_table(result.state))
                record['state']=[dict(note_id=asdict(key),pitch=pitch) for key,pitch in result.state]
            else:
                inherited=PilotObjective(observation,frozen,'logistic93',target)
                config=SearchConfig(population_size=32,generations=16,stagnation_generations=17)
                result,telemetry=run_search(observation.piece,frozen['profiles'][target],inherited,config=config,seed=seed)
                data=original if result.output==observation.piece else MidiPrettyPrinter().to_bytes(result.output)
                history=result.history; gain=result.evaluation.style_gain
                accounting=dict(proposals=512,evaluation_requests=545,unique_evaluations=result.unique_candidates,
                    cache_hits=result.cache_hits,rejection_reasons=dict(inherited.rejections),telemetry=telemetry,stop_reason=result.stop_reason)
                record['genome']=asdict(result.genome)
            (directory/'output.mid').write_bytes(data)
            content=measure_content(observation,observe_midi(data))
            if content_failures(content): raise ValueError('selected output content failure')
            if abs(objective(data)-baseline-gain)>1e-12: raise ValueError('saved optimized affinity mismatch')
            write_json(directory/'history.json',history)
            extraction_result=runtime.extract(directory/'output.mid',directory/'external')
            external_score=score_external(external['external'],ext_metadata['schema'],{'output':extraction_result},['output'])[0]
            before=source_scores[sid]; after=external_score.get('affinities')
            external_move=movement(before,after,row['composer'],target) if before and after else None
            record.update(status='completed',optimized_gain=gain,accounting=accounting,content=content,
                identity=data==original,output_sha256=sha256_file(directory/'output.mid'),external=external_score,
                external_movement=external_move,external_delta=external_move['delta'] if external_move else None)
        except Exception as exc:
            record['failure']=dict(type=type(exc).__name__,message=str(exc))
        record['elapsed_seconds']=time.perf_counter()-started
        results.append(record); write_json(directory/'result.json',record); write_json(output/'results.json',results)
        print(f'{index+1}/48 {method} {record["status"]}',flush=True)
    paired=[]
    for row in cohort:
        for target in composers:
            if target==row['composer']: continue
            for seed in (1729,1730):
                pair={r['method']:r for r in results if r['source_id']==row['sample_id'] and r['target']==target and r['seed']==seed}
                e3_delta=pair['e3_zero'].get('external_delta'); pitch_delta=pair['pitch_search'].get('external_delta')
                paired.append(dict(sample_id=row['sample_id'],group_id=row['group_id'],composer=row['composer'],target=target,seed=seed,
                    pitch_minus_e3=None if e3_delta is None or pitch_delta is None else pitch_delta-e3_delta))
    write_json(output/'paired_comparison.json',paired)
    summary=dict(runs=len(results),completed=sum(r['status']=='completed' for r in results),
        paired_external_difference=cluster_summary(paired,'pitch_minus_e3'),
        by_direction={a+'->'+b:{m:cluster_summary([r for r in results if r['composer']==a and r['target']==b and r['method']==m],'external_delta') for m in ('e3_zero','pitch_search')} for a in composers for b in composers if a!=b},
        by_method={m:cluster_summary([r for r in results if r['method']==m],'external_delta') for m in ('e3_zero','pitch_search')},
        listening_pending=True,automatic_promotion=False,interpretation='technical content tests and optimized classifier gains do not prove musical validity')
    write_json(output/'summary.json',summary)
    return summary


def synthetic_verification(checkout,output,configuration=None):
    from .accompaniment_search import PitchSource,local_search
    from .midi.types import InternalRepr,NoteEvent,MetaEvent
    from .midi.printer import MidiPrettyPrinter
    from .midi.parser import MidiParser
    fresh(checkout,output,configuration)
    cases=[]
    for index,(label,smf,channel,duration) in enumerate((('chords',1,0,240),('format0',0,7,120),('zero_duration',1,2,0))):
        piece=InternalRepr(480,tuple(n for i in range(5) for n in (NoteEvent(i*480,channel,40+i*2,64,duration),NoteEvent(i*480,channel,80,90,300))),
            (MetaEvent(0,'tempo',{'tempo':500000}),MetaEvent(0,'time_signature',{'numerator':3,'denominator':4})),smf)
        original=MidiPrettyPrinter().to_bytes(piece); source=PitchSource.from_bytes(original)
        objective=lambda data:float(sum(n.pitch for n in MidiParser().parse_bytes(data).notes))
        result=local_search(source,objective,proposals=64)
        repeated=local_search(source,objective,proposals=64)
        if result.midi!=repeated.midi or result.history!=repeated.history: raise AssertionError('synthetic repeatability')
        (output/(label+'_before.mid')).write_bytes(original); (output/(label+'_after.mid')).write_bytes(result.midi)
        write_json(output/(label+'_notes.json'),source.note_table(result.state))
        table=['| Protected | Tick | Duration | Velocity | Channel | Before | After |','|---|---:|---:|---:|---:|---:|---:|']
        for row in source.note_table(result.state):
            table.append('| '+' | '.join(str(row[k]) for k in ('protected','tick','duration','velocity','channel','before','after'))+' |')
        (output/(label+'_notes.md')).write_text('\n'.join(table)+'\n',encoding='utf-8')
        cases.append(dict(label=label,repeatability=True,accounting=result.accounting,notes=source.note_table(result.state)))
    summary=dict(passed=True,synthetic_cases=cases,scientific_data_access=False,objective='synthetic arithmetic test double; no scientific style metric')
    write_json(output/'summary.json',summary)
    return summary
