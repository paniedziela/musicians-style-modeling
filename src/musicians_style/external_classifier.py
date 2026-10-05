"""Nullable external descriptors with frozen V2-04 logistic settings, train only."""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from .style_metrics import LOGISTIC, assert_disjoint


def matrix(attempts,ids,schema):
    rows=[]
    for sample in ids:
        record=attempts[sample]
        if record['status']!='success':
            raise ValueError('failed extraction retained outside classifier: '+sample)
        if record['schema']!=schema:
            raise ValueError('external schema mismatch: '+sample)
        if len(record['values'])!=len(schema):
            raise ValueError('external value width mismatch')
        rows.append([np.nan if v is None else float(v) for v in record['values']])
    x=np.asarray(rows,dtype=float)
    if np.isinf(x).any(): raise ValueError('infinite external value')
    return x


def fit_external(train,forbidden,attempts):
    assert_disjoint(train,forbidden)
    successful=[r for r in sorted(train,key=lambda r:r['sample_id']) if attempts[r['sample_id']]['status']=='success']
    failed=[r['sample_id'] for r in train if attempts[r['sample_id']]['status']!='success']
    if not successful or {r['composer'] for r in successful}!={r['composer'] for r in train}:
        raise ValueError('training extraction lacks composer coverage')
    schema=attempts[successful[0]['sample_id']]['schema']
    x=matrix(attempts,[r['sample_id'] for r in successful],schema)
    pipeline=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('variance',VarianceThreshold()),('scale',StandardScaler()),('model',LogisticRegression(**LOGISTIC))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pipeline.fit(x,[r['composer'] for r in successful])
    metadata=dict(schema=schema,train=train,forbidden=forbidden,failed_training_ids=failed,
        successful_training_ids=[r['sample_id'] for r in successful],parameters=LOGISTIC,
        imputation='train median; entirely missing train columns become zero then variance-excluded',
        imputer_statistics=pipeline[0].statistics_.tolist(),variance_support=pipeline[1].get_support().tolist(),
        scaler_mean=pipeline[2].mean_.tolist(),scaler_scale=pipeline[2].scale_.tolist(),
        classes=pipeline[-1].classes_.tolist(),coef=pipeline[-1].coef_.tolist(),
        intercept=pipeline[-1].intercept_.tolist(),iterations=pipeline[-1].n_iter_.tolist(),
        warnings=[str(w.message) for w in caught])
    return pipeline,metadata


def score_external(pipeline,schema,attempts,ids):
    rows=[]
    for sample in ids:
        record=attempts[sample]
        if record['status']!='success':
            rows.append(dict(sample_id=sample,status='failure',failure=record['failure']))
            continue
        try:
            values=pipeline.predict_proba(matrix(attempts,[sample],schema))[0]
            rows.append(dict(sample_id=sample,status='success',affinities=dict(zip(pipeline.classes_,map(float,values)))))
        except Exception as exc:
            rows.append(dict(sample_id=sample,status='failure',failure=dict(type=type(exc).__name__,message=str(exc))))
    return rows
