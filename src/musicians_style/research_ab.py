"""Explicit jSymbolic feasibility and train-only external evaluation workflows."""
from __future__ import annotations
import json
from pathlib import Path

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
