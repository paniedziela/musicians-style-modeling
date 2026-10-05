import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import numpy as np
import pytest

from musicians_style.feature_backends.jsymbolic import parse_exports,numeric_schema
from musicians_style.external_classifier import fit_external,score_external
from musicians_style.jsymbolic_runtime import JSymbolicRuntime
from musicians_style.provenance import sha256_file


def ace(tmp_path,*,dimension=2,values='<v>1</v><v>NaN</v>',rows=1):
    defs=tmp_path/'defs.xml'; data=tmp_path/'data.xml'
    defs.write_text(f'<feature_key_file><feature><name>Pitch</name><parallel_dimensions>{dimension}</parallel_dimensions></feature><feature><name>Instrument</name><parallel_dimensions>1</parallel_dimensions></feature></feature_key_file>')
    row=f'<data_set><data_set_id>COMPOSER_LABEL/path</data_set_id><feature><name>Instrument</name><v>1</v></feature><feature><name>Pitch</name>{values}</feature></data_set>'
    data.write_text('<feature_vector_file>'+row*rows+'</feature_vector_file>')
    return data,defs


def test_numeric_projection_nulls_no_ids_labels_instrumentation(tmp_path):
    values,defs=ace(tmp_path)
    result=parse_exports(values,defs,[dict(name='Pitch',dimensions=2)])
    assert result['values']==[1.,None]
    assert result['diagnostics']['excluded']==['Instrument']
    assert [f['name'] for f in result['schema']]==['Pitch[000]','Pitch[001]']
    assert parse_exports(values,defs,[dict(name='Pitch',dimensions=2)])==result


@pytest.mark.parametrize('dimension,values,rows,reason',[(3,'<v>1</v><v>2</v>',1,'schema'),(2,'<v>1</v>',1,'dimension'),(2,'<v>composer</v><v>1</v>',1,'float'),(2,'<v>1</v><v>2</v>',0,'exactly one'),(2,'<v>1</v><v>2</v>',2,'exactly one')])
def test_schema_failures_are_explicit(tmp_path,dimension,values,rows,reason):
    data,defs=ace(tmp_path,dimension=dimension,values=values,rows=rows)
    with pytest.raises(ValueError,match=reason): parse_exports(data,defs,[dict(name='Pitch',dimensions=2)])


def row(i,composer): return dict(sample_id=str(i),group_id='work'+str(i),sha256='hash'+str(i),composer=composer)


def training():
    train=[row(i,'ABC'[i%3]) for i in range(9)]
    schema=numeric_schema([dict(name='Pitch',dimensions=3)])
    attempts={r['sample_id']:dict(status='success',schema=schema,values=[float(i),None if i==0 else i%3,1.]) for i,r in enumerate(train)}
    forbidden=[row(10,'A')]
    attempts['10']=dict(status='success',schema=schema,values=[1e9,-1e9,10.])
    return train,forbidden,attempts


def test_training_only_imputation_selection_scaling_and_fixed_classifier():
    train,forbidden,attempts=training()
    fitted,meta=fit_external(train,forbidden,attempts)
    attempts['10']['values']=[-1e20,1e20,10000.]
    other,other_meta=fit_external(train,forbidden,attempts)
    assert meta==other_meta
    assert fitted[0].statistics_.tolist()==[4.,1.,1.]
    assert fitted[1].get_support().tolist()==[True,True,False]
    assert np.array_equal(fitted[-1].coef_,other[-1].coef_)
    assert fitted[-1].C==1 and fitted[-1].class_weight=='balanced'
    assert fitted[-1].solver=='lbfgs' and fitted[-1].max_iter==5000


@pytest.mark.parametrize('key',['sample_id','group_id','sha256'])
def test_leakage_guard_before_learned_operations(key):
    train,forbidden,attempts=training()
    forbidden[0][key]=train[0][key]
    with pytest.raises(ValueError,match=key): fit_external(train,forbidden,attempts)


def test_failures_stay_in_denominators_and_missing_training_columns():
    train,forbidden,attempts=training()
    attempts['0']=dict(status='failure',failure=dict(type='TestFailure',message='x'))
    for r in train[1:]: attempts[r['sample_id']]['values'][1]=None
    fitted,metadata=fit_external(train,forbidden,attempts)
    assert metadata['failed_training_ids']==['0']
    assert fitted[1].get_support().tolist()==[True,False,False]
    scores=score_external(fitted,metadata['schema'],attempts,['0','1'])
    assert len(scores)==2 and scores[0]['status']=='failure' and scores[1]['status']=='success'
    assert len(metadata['train'])==9


@pytest.mark.parametrize('failure',['timeout','exit','zero_exit_no_rows','pin','corrupt'])
def test_runtime_failure_records(tmp_path,monkeypatch,failure):
    root=Path(__file__).resolve().parents[2]
    runtime=JSymbolicRuntime(root,tmp_path/'distribution',tmp_path/'java.exe')
    if failure=='pin':
        def verify(): raise ValueError('pin mismatch')
    else:
        verify=lambda:dict(java_version='fake')
    monkeypatch.setattr(runtime,'verify',verify)
    source=tmp_path/'source.mid'; source.write_bytes(b'test')
    monkeypatch.setattr('musicians_style.jsymbolic_runtime.sanitize_midi',lambda a,b:b.write_bytes(b'test'))
    def process(command,**kwargs):
        if failure=='timeout': raise subprocess.TimeoutExpired(command,1)
        if failure=='exit': return SimpleNamespace(returncode=7)
        if failure=='corrupt':
            (tmp_path/'out/values.xml').write_text('not XML')
            (tmp_path/'out/definitions.xml').write_text('not XML')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('musicians_style.jsymbolic_runtime.subprocess.run',process)
    result=runtime.extract(source,tmp_path/'out')
    assert result['status']=='failure' and result['failure']['type']
    assert json.loads((tmp_path/'out/result.json').read_text())['status']=='failure'
    with pytest.raises(ValueError,match='fresh'): runtime.extract(source,tmp_path/'out')


def test_actual_pinned_schema_and_configuration_exclusions():
    root=Path(__file__).resolve().parents[2]
    lock=json.loads((root/'requirements/research-ab-jsymbolic.lock.json').read_text())
    assert sha256_file(root/lock['config'])==lock['config_sha256']
    assert sha256_file(root/lock['schema'])==lock['schema_sha256']
    schema=json.loads((root/lock['schema']).read_text())
    assert len(schema)==42 and len(numeric_schema(schema))==638
    assert not any(any(word in f['name'].lower() for word in ('instrument','program','composer','filename','velocity','microtone','vibrato','glissando')) for f in schema)
