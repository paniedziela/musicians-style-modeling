from collections import Counter
from dataclasses import FrozenInstanceError
import io
import json

import mido
import numpy as np
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from musicians_style.accompaniment_search import PitchSource, MoveRejected, local_search, Logistic93Objective
from musicians_style.midi.parser import MidiParser
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.midi.types import InternalRepr, NoteEvent, MetaEvent
from musicians_style.e3.structure import analyse_structure
from musicians_style.e3.transformation import apply_transformation
from musicians_style.e3.types import E3Genome
from musicians_style.e3.profile import build_target_profile
from musicians_style.e1.composition_features import extract_composition_features
from musicians_style.style_metrics import LOGISTIC


def synthetic(pitches=(40,42,44,46,48), *, smf=1, channel=0, zero=False):
    notes=[]
    for i,pitch in enumerate(pitches):
        notes += [NoteEvent(i*480,channel,pitch,64,0 if zero else 200),
                  NoteEvent(i*480,channel,80,90,300)]
    piece=InternalRepr(480,tuple(notes),(MetaEvent(0,'tempo',{'tempo':500000}),),smf)
    midi=mido.MidiFile(file=io.BytesIO(MidiPrettyPrinter().to_bytes(piece)))
    midi.tracks[-1].insert(0,mido.Message('control_change',channel=channel,control=64,value=127,time=0))
    midi.tracks[0].insert(0,mido.MetaMessage('track_name',name='untouched',time=0))
    output=io.BytesIO(); midi.save(file=output)
    return output.getvalue()


def raw(data):
    midi=mido.MidiFile(file=io.BytesIO(data))
    return [[msg.dict() for msg in track] for track in midi.tracks]


@pytest.mark.parametrize('smf,channel,zero',[(0,0,False),(1,7,False),(1,0,True)])
def test_pitch_only_and_exact_raw_roundtrip(smf,channel,zero):
    source=PitchSource.from_bytes(synthetic(smf=smf,channel=channel,zero=zero))
    assert source.serialize() == source.original
    key=source.accompaniment[0]
    state,output=source.move((),key,key.pitch+4)
    before,after=raw(source.original),raw(output)
    changed=[]
    for ti,(left,right) in enumerate(zip(before,after)):
        for i,(a,b) in enumerate(zip(left,right)):
            if a != b:
                assert a['type'] in ('note_on','note_off')
                assert dict(a,note=a['note']+4)==b
                changed.append((ti,i))
    assert len(changed)==2
    assert source.protected==PitchSource.from_bytes(source.original).protected
    assert len(MidiParser().parse_bytes(output).notes)==len(source.piece.notes)
    assert source.note_table(state)[0]['after']==44


def test_source_mask_metadata_and_state_immutable():
    source=PitchSource.from_bytes(synthetic())
    with pytest.raises(FrozenInstanceError): source.original=b'bad'
    piece=source.piece
    piece.meta[0].payload['tempo']=1
    assert source.piece.meta[0].payload['tempo']==500000
    key=source.accompaniment[0]
    input_map={key:44}
    state=source.canonical(input_map)
    input_map[key]=45
    assert dict(state)[key]==44


@pytest.mark.parametrize('pitch,reason',[(-1,'pitch_range'),(128,'pitch_range'),(45,'original_pitch_radius'),(40.0,'integer_pitch'),(True,'integer_pitch')])
def test_reject_bad_pitch(pitch,reason):
    source=PitchSource.from_bytes(synthetic())
    with pytest.raises(MoveRejected,match=reason): source.move((),source.accompaniment[0],pitch)


def test_protection_and_cumulative_ceiling():
    source=PitchSource.from_bytes(synthetic())
    with pytest.raises(MoveRejected,match='protected'): source.move((),next(iter(source.protected)),79)
    first,second=source.accompaniment[:2]
    state,_=source.move((),first,44)
    with pytest.raises(MoveRejected,match='ceiling'): source.move(state,second,46)
    reverted,_=source.move(state,first,40)
    assert reverted==()
    with pytest.raises(MoveRejected,match='radius'): source.move(state,first,48)


def test_small_set_floor_identity_and_unknown_duplicate_ids():
    source=PitchSource.from_bytes(synthetic((40,)))
    key=source.accompaniment[0]
    with pytest.raises(MoveRejected,match='ceiling'): source.move((),key,41)
    with pytest.raises(MoveRejected,match='duplicate'): source.serialize(((key,41),(key,42)),changed_fraction=1)
    with pytest.raises(MoveRejected,match='unknown'): source.serialize(((object(),41),),changed_fraction=1)
    assert source.serialize()==source.original


def test_collision_cross_channel_and_protected_overtop():
    piece=InternalRepr(480,(NoteEvent(0,0,60,60,400),NoteEvent(0,0,62,60,400),NoteEvent(0,0,63,80,400)),(),0)
    source=PitchSource.from_bytes(MidiPrettyPrinter().to_bytes(piece))
    key=next(key for key in source.accompaniment if key.pitch==60)
    with pytest.raises(MoveRejected,match='overlap'): source.move((),key,62,changed_fraction=1)
    with pytest.raises(MoveRejected,match='overtops'): source.move((),key,64,changed_fraction=1)
    # Moving into an existing pitch on another channel is valid.
    piece=InternalRepr(480,(NoteEvent(0,0,60,60,400),NoteEvent(0,1,62,60,400),NoteEvent(0,0,80,80,400)),(),0)
    source=PitchSource.from_bytes(MidiPrettyPrinter().to_bytes(piece))
    key=next(key for key in source.accompaniment if key.pitch==60)
    source.move((),key,62,changed_fraction=1)


def test_sustained_accompaniment_overtopping_later_attack():
    piece=InternalRepr(480,(NoteEvent(0,0,60,60,800),NoteEvent(0,0,80,80,400),NoteEvent(480,0,62,80,400)),(),0)
    source=PitchSource.from_bytes(MidiPrettyPrinter().to_bytes(piece))
    with pytest.raises(MoveRejected,match='overtops'): source.move((),source.accompaniment[0],64,changed_fraction=1)


def test_overlaps_duplicates_cross_tracks_and_zero_off():
    midi=mido.MidiFile(type=1,ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([mido.MetaMessage('time_signature',numerator=3,denominator=4)]))
    for ch in (0,1):
        track=mido.MidiTrack([mido.Message('note_on',channel=ch,note=40,velocity=60,time=0),
            mido.Message('note_on',channel=ch,note=80,velocity=80,time=0),
            mido.Message('note_on',channel=ch,note=40,velocity=60,time=100),
            mido.Message('note_on',channel=ch,note=40,velocity=0,time=100),
            mido.Message('note_off',channel=ch,note=40,velocity=9,time=100),
            mido.Message('note_off',channel=ch,note=80,time=100)])
        midi.tracks.append(track)
    buf=io.BytesIO(); midi.save(file=buf)
    source=PitchSource.from_bytes(buf.getvalue())
    assert source.serialize()==buf.getvalue()
    key=next(key for key in source.accompaniment if key.pitch==40)
    _,out=source.move((),key,39,changed_fraction=1)
    assert len(MidiParser().parse_bytes(out).notes)==6
    assert len(source.protected)==2


def test_exact_duplicate_ids_are_distinct():
    piece=InternalRepr(480,(NoteEvent(0,0,40,60,100),NoteEvent(0,0,40,60,100),NoteEvent(0,0,80,80,100)),(),0)
    source=PitchSource.from_bytes(MidiPrettyPrinter().to_bytes(piece))
    assert {key.duplicate for key in source.accompaniment}=={0,1}
    source.move((),source.accompaniment[1],41,changed_fraction=1)


def test_unmatched_source_identity_fallback():
    midi=mido.MidiFile(type=0)
    midi.tracks.append(mido.MidiTrack([mido.Message('note_on',note=40,velocity=60),mido.Message('note_on',note=80,velocity=80),mido.MetaMessage('end_of_track',time=200)]))
    buf=io.BytesIO(); midi.save(file=buf)
    source=PitchSource.from_bytes(buf.getvalue())
    assert not source.editable and source.serialize()==buf.getvalue()
    with pytest.raises(MoveRejected,match='unmatched'): source.move((),source.accompaniment[0],41,changed_fraction=1)



def test_search_determinism_budget_visited_and_identity():
    source=PitchSource.from_bytes(synthetic())
    def objective(data): return sum(n.pitch for n in MidiParser().parse_bytes(data).notes)
    a=local_search(source,objective,proposals=64)
    b=local_search(source,objective,proposals=64)
    assert a.state==b.state and a.midi==b.midi and a.history==b.history
    assert a.accounting['proposals']==64 and len(a.history)==64
    assert json.loads(json.dumps(a.history))[0]['status']=='identity'
    assert a.accounting['cache_hits']>0 and a.accounting['rejections']>0
    assert len(a.state)<=1 and a.gain>0
    assert sum(a.accounting[k] for k in ('unique_evaluations','cache_hits','rejections'))==64
    falling=local_search(source,lambda data:-objective(data),proposals=64)
    assert falling.affinity>=-objective(source.original)
    constant=local_search(source,lambda data:1,proposals=64)
    assert constant.midi==source.original and constant.state==() and constant.gain==0


def test_logistic_frozen_objective_exact_probability():
    x=np.random.default_rng(1729).normal(size=(12,93))
    fitted=Pipeline([('variance',VarianceThreshold()),('scale',StandardScaler()),('model',LogisticRegression(**LOGISTIC))]).fit(x,['A','B','C']*4)
    data=synthetic()
    objective=Logistic93Objective(fitted,'B')
    expected=fitted.predict_proba(extract_composition_features(MidiParser().parse_bytes(data)).reshape(1,-1))[0,1]
    assert objective(data)==expected
    fitted[-1].coef_[:]=0
    assert objective(data)==expected


def test_frozen_e3_historical_transposition_differential():
    source=PitchSource.from_bytes(synthetic())
    piece=source.piece
    rows=[dict(sample_id='train',group_id='work',sha256='hash',composer='A')]
    profile=build_target_profile('A',rows,{'train':piece},forbidden_rows=[])
    transformed=apply_transformation(piece,E3Genome(transpose_semitones=2),profile,seed=1729)
    assert Counter(transformed.notes)==Counter(NoteEvent(n.tick,n.channel,n.pitch+2,n.velocity,n.duration_ticks) for n in piece.notes)
    assert apply_transformation(piece,E3Genome(),profile,seed=1729)==piece
    # New adapter permits exactly one accompaniment pitch, zero protected transposition.
    _,output=source.move((),source.accompaniment[0],42)
    assert sum(n.pitch for n in MidiParser().parse_bytes(output).notes)-sum(n.pitch for n in piece.notes)==2



def test_stable_ids_and_tie_mask_use_original_only():
    notes=(NoteEvent(0,1,80,70,300),NoteEvent(0,0,80,80,200),NoteEvent(0,0,40,64,100))
    left=analyse_structure(InternalRepr(480,notes,(),0))
    right=analyse_structure(InternalRepr(480,tuple(reversed(notes)),(),0))
    assert left.notes==right.notes and left.melody_ids==right.melody_ids
    assert len(left.melody_ids)==1 and left.ambiguous_melody_onsets==(0,)


def test_serialization_failure_is_rejection_and_identity_stays_original(monkeypatch):
    source=PitchSource.from_bytes(synthetic())
    def fail(*args,**kwargs): raise ValueError('injected serializer failure')
    monkeypatch.setattr('musicians_style.accompaniment_search.mido.MidiFile.save',fail)
    result=local_search(source,lambda data:1.,proposals=16)
    assert result.state==() and result.midi==source.original
    assert any('serializer failure' in row.get('reason','') for row in result.history)


def test_overlap_new_pitch_against_another_track_rejected():
    midi=mido.MidiFile(type=1)
    for pitch in (40,42,80):
        midi.tracks.append(mido.MidiTrack([mido.Message('note_on',note=pitch,velocity=64),mido.Message('note_off',note=pitch,time=100)]))
    buf=io.BytesIO(); midi.save(file=buf)
    source=PitchSource.from_bytes(buf.getvalue())
    key=next(key for key in source.accompaniment if key.pitch==40)
    with pytest.raises(MoveRejected,match='overlap'): source.move((),key,42,changed_fraction=1)
