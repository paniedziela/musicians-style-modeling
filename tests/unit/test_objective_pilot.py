from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import pytest
from musicians_style.e3 import algorithm
from musicians_style.e3.algorithm import SearchConfig
from musicians_style.e3.types import E3Genome
from musicians_style.midi.types import InternalRepr, NoteEvent
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.content_metrics import observe_midi, measure_content
from musicians_style.e1.composition_features import extract_composition_features
from musicians_style.objective_pilot import (SEARCH, OBJECTIVES, PROTOCOL, PilotObjective, content_failures,
    objective_affinity, run_search, select_cohort, summarize)
from musicians_style.style_metrics import fit_measures, event_profiles, score_measures
from musicians_style.e3.profile import style_vector

@pytest.fixture
def fitted():
    rows, reprs = [], {}
    for ci,c in enumerate(('A','B','C')):
        for i in range(3):
            sid=f'{c}{i}'
            rows.append(dict(sample_id=sid,group_id=sid,sha256=sid,composer=c))
            reprs[sid]=InternalRepr(480, tuple(NoteEvent(t,0,48+ci*5+p,80,d+i*10) for t,p,d in
                [(0,0,120),(0,12,240),(480,2,120),(480,14,240),(960,4,120),(960,16,240)]))
    custom={s:extract_composition_features(p) for s,p in reprs.items()}
    events={s:event_profiles(p)[0] for s,p in reprs.items()}
    bundle,_=fit_measures(rows,[],reprs,custom,events)
    return bundle,reprs['A0']

@pytest.mark.parametrize('name',OBJECTIVES)
def test_adapter_zero_before_transform_and_evaluate_deterministic_and_no_global_change(fitted,name):
    bundle,source=fitted
    obs=observe_midi(MidiPrettyPrinter().to_bytes(source))
    original_clamp,original_transform=algorithm._clamp,algorithm.apply_transformation
    cfg=SearchConfig(population_size=4,generations=3,elitism_k=1,tournament_size=3,
                     stagnation_generations=4,mutation_sigma=(100,.12,.12,.12,.12))
    a,ta=run_search(source,bundle['profiles']['B'],PilotObjective(obs,bundle,name,'B'),cfg)
    b,tb=run_search(source,bundle['profiles']['B'],PilotObjective(obs,bundle,name,'B'),cfg)
    assert a==b and ta==tb and ta['nonzero_gene_discarded']>0
    assert a.genome.transpose_semitones==0
    assert all(r['best_genome']['transpose_semitones']==0 for r in a.history)
    assert a.unique_candidates+a.cache_hits==1+4*4
    assert algorithm._clamp is original_clamp and algorithm.apply_transformation is original_transform
    assert not content_failures(measure_content(obs,observe_midi(MidiPrettyPrinter().to_bytes(a.output))))

@pytest.mark.parametrize('name',OBJECTIVES)
def test_objective_adaptation_matches_reused_style_measure(fitted,name):
    bundle,piece=fitted
    all_scores=score_measures(bundle,style_vector(piece)[0],extract_composition_features(piece),event_profiles(piece)[0])
    assert objective_affinity(bundle,name,'B',piece)==all_scores[name]['B']
    with pytest.raises(AssertionError,match='transposition'):
        PilotObjective(observe_midi(MidiPrettyPrinter().to_bytes(piece)),bundle,name,'B').evaluate(E3Genome(1),piece)


def test_same_tick_order_diagnostic_duration_failure_hard(fitted):
    _,source=fitted
    obs=observe_midi(MidiPrettyPrinter().to_bytes(source))
    metric=measure_content(obs,obs)
    metric['v2_exact_pitch']['literal_event_order_status']='failed'
    metric['v2_exact_pitch']['event_order_status']='failed'
    metric['v2_exact_pitch']['status']='failed'
    assert not content_failures(metric)
    metric['v2_exact_pitch']['duration_status']='ambiguous'
    assert not content_failures(metric)
    metric['v2_exact_pitch']['duration_status']='failed'
    assert 'protected_duration' in content_failures(metric)


def test_fixed_budget_and_no_event_objectives():
    assert (SEARCH.population_size,SEARCH.generations,SEARCH.stagnation_generations)==(32,60,61)
    assert SEARCH.mutation_sigma==(1,.12,.12,.12,.12)
    assert PROTOCOL['budget']['proposal_budget']==1832
    assert PROTOCOL['budget']['evaluation_requests_including_elites_and_identity']==1953
    assert OBJECTIVES==('rms67','gaussian67','logistic93')
    assert summarize([])['rms67']['measures']['time_pitch']['mean'] is None


def test_lexical_cohort_selection_does_not_use_note_counts_or_scores():
    rows=[dict(sample_id=f'{c}{i}',group_id=f'{c}{i}',sha256=f'{c}{i}',composer=c,note_count=100-i)
          for c in ('A','B','C') for i in range(3)]
    folds=[]
    for f in range(3):
        test=[r['sample_id'] for r in rows if r['sample_id'].endswith(str(f))]
        train=[r['sample_id'] for r in rows if r['sample_id'] not in test]
        a=[s for s in train if s.endswith(str((f+1)%3))]
        b=[s for s in train if s not in a]
        folds.append(dict(fold=f,train=dict(sample_ids=train),test=dict(sample_ids=test),inner_folds=[
            dict(fold=0,train=dict(sample_ids=a),validation=dict(sample_ids=b)),
            dict(fold=1,train=dict(sample_ids=b),validation=dict(sample_ids=a))]))
    splits=dict(repetitions=[dict(repeat=0,folds=folds)])
    cohort,train,forbidden=select_cohort(rows,splits)
    assert [r['sample_id'] for r in cohort]==['A2','B2','C2']
    changed=[dict(r,note_count=-999,style_score=999) for r in reversed(rows)]
    assert [r['sample_id'] for r in select_cohort(changed,splits)[0]]==['A2','B2','C2']
    assert not {r['group_id'] for r in train}&{r['group_id'] for r in forbidden}

@pytest.mark.integration
def test_complete_synthetic_runner_persists_fits_before_search_and_never_opens_test_sources(tmp_path,monkeypatch):
    import musicians_style.objective_pilot as pilot
    from musicians_style.e3.algorithm import SearchResult
    from musicians_style.e3.types import CandidateEvaluation
    root=tmp_path/'checkout'
    data=root/'datasets'
    derived=data/'derived/e1_asap'
    source_root=data/'asap-dataset-1.2'
    derived.mkdir(parents=True)
    source_root.mkdir()
    (root/'experiments').mkdir()
    for relative in ('src/musicians_style/objective_pilot.py','tools/objective_pilot.py','docs/research/V2_05_PROTOCOL.md'):
        p=root/relative
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('fixture')
    rows=[]
    for ci,c in enumerate(('A','B','C')):
        for i in range(3):
            sid=f'{c}{i}'
            notes=tuple(NoteEvent(t,0,48+ci*5+p,80,d+i*10) for t,p,d in [(0,0,120),(0,12,240),(480,2,120),(480,14,240)])
            path=source_root/f'{sid}.mid'
            path.write_bytes(MidiPrettyPrinter().to_bytes(InternalRepr(480,notes)))
            rows.append(dict(sample_id=sid,group_id=sid,sha256=pilot.sha256_file(path),composer=c,score_path=path.name,validation_status='accepted'))
    folds=[]
    for f in range(3):
        test=[r['sample_id'] for r in rows if r['sample_id'].endswith(str(f))]
        train=[r['sample_id'] for r in rows if r['sample_id'] not in test]
        a=[s for s in train if s.endswith(str((f+1)%3))]
        b=[s for s in train if s not in a]
        folds.append(dict(fold=f,train=dict(sample_ids=train),test=dict(sample_ids=test),inner_folds=[
            dict(fold=0,train=dict(sample_ids=a),validation=dict(sample_ids=b)),
            dict(fold=1,train=dict(sample_ids=b),validation=dict(sample_ids=a))]))
    (derived/'manifest.json').write_text(json.dumps(dict(samples=rows)))
    (derived/'splits.json').write_text(json.dumps(dict(repetitions=[dict(repeat=0,folds=folds)])))
    opened=[]
    observe=pilot.observe_midi
    def guarded_observe(path):
        if isinstance(path,Path):
            assert not path.stem.endswith('0'), 'outer test was opened'
            opened.append(path.stem)
        return observe(path)
    monkeypatch.setattr(pilot,'observe_midi',guarded_observe)
    dest=root/'experiments/pilot'
    def identity_search(source,profile,objective):
        assert (dest/'pilot_manifest.json').exists()
        assert (dest/'fit_manifest.json').exists()
        assert (dest/'optimization_started.json').exists()
        value=objective.evaluate(E3Genome(),source)
        assert value.constraints.feasible
        result=SearchResult(E3Genome(),source,value,tuple(dict(generation=i,best_genome=dict(transpose_semitones=0)) for i in range(61)),'max_generations',1,1952)
        return result,dict(transformations=1)
    monkeypatch.setattr(pilot,'run_search',identity_search)
    audit=pilot.run_pilot(root,dest,provenance={})
    assert audit['passed'] and audit['runs']==18
    assert set(opened)=={'A1','A2','B1','B2','C1','C2'}
    outputs=json.loads((dest/'per_output.json').read_text())
    assert all(r['optimized_gain']==0 and r['source_sha256']==r['output_sha256'] for r in outputs)
    assert all(r['movement']['time_pitch']['delta']==0 for r in outputs)
    with pytest.raises(ValueError,match='fresh'):
        pilot.run_pilot(root,dest)
