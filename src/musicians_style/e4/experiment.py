"""Reproducible E4.3 harness: preparation, training and full-score inference."""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from ..midi.parser import MidiParser
from ..midi.printer import MidiPrettyPrinter
from .dataset import (
    COMPOSERS,
    BalancedComposerSampler,
    PieceSegments,
    SegmentDataset,
    assert_split_isolation,
)
from .model import ConditionalGenerator, PatchDiscriminator
from .segmentation import (
    PITCH_HIGH,
    PITCH_LOW,
    STEPS_PER_BAR,
    Segment,
    encode_piece,
    quantization_errors,
    stitch_segments,
)
from .training import gan_step

SCHEMA_VERSION = "e4.3.0"

@dataclass(frozen=True)
class E4Config:
    manifest_path: Path | None
    splits_path: Path | None
    dataset_root: Path | None
    run_dir: Path | None
    repeat: int = 0
    outer_fold: int = 0
    inner_fold: int = 0
    seed: int = 1729
    batch_size: int = 16
    max_epochs: int = 30
    smoke_epochs: int = 2
    smoke_per_composer: int = 4
    smoke_files: int = 6
    conv_dim: int = 32
    residual_blocks: int = 3
    device: str = "auto"


# Configuration and reproducibility -------------------------------------------------

def _resolve(value: str) -> Path:
    path = Path(value); return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
def load_e4_config(path: Path | str) -> E4Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != SCHEMA_VERSION: raise ValueError(f"unsupported E4 config; expected {SCHEMA_VERSION}")
    inputs, output, training, smoke, model = (raw.get(k, {}) for k in ("inputs", "output", "training", "smoke", "model"))
    return E4Config(_resolve(str(inputs["manifest"])), _resolve(str(inputs["splits"])), _resolve(str(inputs["dataset_root"])), _resolve(str(output["run_dir"])), int(raw.get("repeat", 0)), int(raw.get("outer_fold", 0)), int(raw.get("inner_fold", 0)), int(raw.get("seed",1729)), int(training.get("batch_size",16)), int(training.get("max_epochs",30)), int(smoke.get("epochs",2)), int(smoke.get("per_composer",4)), int(smoke.get("files",6)), int(model.get("conv_dim",32)), int(model.get("residual_blocks",3)), str(training.get("device","auto")))
def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(path.name + ".tmp"); temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n",encoding="utf-8"); os.replace(temporary,path)
def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(payload,ensure_ascii=False,sort_keys=True)+"\n"); handle.flush(); os.fsync(handle.fileno())
def _utc() -> str: return datetime.now(timezone.utc).isoformat()
def _commit() -> str:
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return "unknown"

def _split_ids(splits: Mapping[str, Any], config: E4Config) -> dict[str,set[str]]:
    repeat=next((x for x in splits["repetitions"] if int(x["repeat"])==config.repeat),None); outer=next((x for x in (repeat or {}).get("folds",[]) if int(x["fold"])==config.outer_fold),None); inner=next((x for x in (outer or {}).get("inner_folds",[]) if int(x["fold"])==config.inner_fold),None)
    if not repeat or not outer or not inner: raise ValueError("requested E4 repeat/fold is missing")
    result={"train":set(inner["train"]["sample_ids"]),"validation":set(inner["validation"]["sample_ids"]),"test":set(outer["test"]["sample_ids"])}
    if any(result[a]&result[b] for a in result for b in result if a!=b): raise ValueError("sample_id leakage between E4 splits")
    return result
def _inputs(config:E4Config) -> tuple[dict[str,Any],dict[str,set[str]],dict[str,dict[str,Any]]]:
    assert config.manifest_path and config.splits_path
    manifest=json.loads(config.manifest_path.read_text(encoding="utf-8")); partitions=_split_ids(json.loads(config.splits_path.read_text(encoding="utf-8")),config); rows={str(x["sample_id"]):x for x in manifest.get("samples",[]) if x.get("validation_status")=="accepted"}
    if set().union(*partitions.values())!=set(rows): raise ValueError("E4 splits do not partition accepted manifest samples")
    groups={name:{str(rows[x]["group_id"]) for x in ids} for name,ids in partitions.items()}
    if any(groups[a]&groups[b] for a in groups for b in groups if a!=b): raise ValueError("group_id leakage between E4 splits")
    return manifest,partitions,rows
def _geometry(c:E4Config)->dict[str,Any]: return {"bars_per_segment":4,"steps_per_bar":STEPS_PER_BAR,"steps_per_segment":64,"pitch_low":PITCH_LOW,"pitch_high":PITCH_HIGH,"channels":["onset","frame"],"composers":list(COMPOSERS),"conv_dim":c.conv_dim,"residual_blocks":c.residual_blocks}


# Audit and preparation --------------------------------------------------------------

def audit_e4(config:E4Config)->Path:
    _, partitions, rows=_inputs(config); assert config.run_dir and config.dataset_root; parser=MidiParser(); report:dict[str,Any]={"schema_version":SCHEMA_VERSION,"geometry":_geometry(config),"splits":{},"rejections":[],"unusual_bar_lengths_quarter_notes":[]}; pitches=[]
    for split,ids in partitions.items():
        counters:dict[str,Counter[str]]=defaultdict(Counter); errors:dict[str,list[tuple[float,float]]]=defaultdict(list)
        for sample_id in sorted(ids):
            row=rows[sample_id]; composer=str(row["composer"])
            try: source=parser.parse(config.dataset_root/row["score_path"]); mapped=encode_piece(source)
            except Exception as exc: report["rejections"].append({"sample_id":sample_id,"reason":f"parser_or_segmentation: {exc}"}); continue
            v=counters[composer]; v["pieces"]+=1; v["bars"]+=len(mapped.bars); v["segments"]+=len(mapped.segments); v["empty_segments"]+=sum(not s.nonempty for s in mapped.segments); v["nonempty_segments"]+=sum(s.nonempty for s in mapped.segments); v["active_cells"]+=sum(int(s.data[1].sum()) for s in mapped.segments); v["onset_cells"]+=sum(int(s.data[0].sum()) for s in mapped.segments)
            for bar in mapped.bars:
                v[f"meter:{bar.numerator}/{bar.denominator}"]+=1; length=bar.length_ticks/source.ticks_per_beat
                if not .5<=length<=12: report["unusual_bar_lengths_quarter_notes"].append({"sample_id":sample_id,"bar":bar.index,"length":length})
            for note in source.notes: pitches.append(note.pitch); v["outside_pitch"]+=int(not PITCH_LOW<=note.pitch<PITCH_HIGH)
            for meter,onset,end in quantization_errors(mapped): errors[meter].append((abs(onset),abs(end)))
        per={c:dict(v) for c,v in sorted(counters.items())}
        for v in per.values():
            steps=max(1,v["segments"]*64); v["occupancy"]=v["active_cells"]/(steps*84); v["onset_density"]=v["onset_cells"]/steps
        report["splits"][split]=per; report.setdefault("quantization",{})[split]={m:{"count":len(v),"onset_abs_p95":sorted(x[0] for x in v)[int(.95*(len(v)-1))],"end_abs_max":max(x[1] for x in v)} for m,v in errors.items()}
    enough=all(report["splits"]["train"].get(c,{}).get("nonempty_segments",0)>=100 for c in COMPOSERS); report["pitch"]={"min":min(pitches,default=None),"max":max(pitches,default=None)}; report["gate"]={"group_isolation":True,"parser_errors":not report["rejections"],"at_least_100_train_nonempty_per_composer":enough,"passed":not report["rejections"] and enough}; target=config.run_dir/"audit.json"; _atomic_json(target,report); return target

def _pieces(config:E4Config,partitions:Mapping[str,set[str]],rows:Mapping[str,Mapping[str,Any]])->dict[str,list[PieceSegments]]:
    assert config.dataset_root; parser=MidiParser(); result={x:[] for x in partitions}; membership={sample:split for split,samples in partitions.items() for sample in samples}
    for sample_id in sorted(membership):
        row=rows[sample_id]; split=membership[sample_id]; result[split].append(PieceSegments(sample_id,str(row["group_id"]),str(row["composer"]),split,encode_piece(parser.parse(config.dataset_root/row["score_path"]))))
    return result
def prepare_e4(config:E4Config)->Path:
    assert config.run_dir and config.manifest_path and config.splits_path; audit=audit_e4(config)
    if not json.loads(audit.read_text(encoding="utf-8"))["gate"]["passed"]: raise ValueError("E4 audit gate failed; refusing to prepare training")
    _,parts,rows=_inputs(config); pieces=_pieces(config,parts,rows); config.run_dir.mkdir(parents=True,exist_ok=True); shutil.copyfile(config.manifest_path,config.run_dir/"manifest.snapshot.json"); shutil.copyfile(config.splits_path,config.run_dir/"splits.snapshot.json")
    snapshot=asdict(config); _atomic_json(config.run_dir/"config.snapshot.json",{k:str(v) if isinstance(v,Path) else v for k,v in snapshot.items()}); segment_path=config.run_dir/"segments.jsonl"; segment_path.unlink(missing_ok=True)
    for split,collection in pieces.items():
        for item in collection: _append_jsonl(segment_path,{"sample_id":item.sample_id,"group_id":item.group_id,"composer":item.composer,"split":split,"segments":len(item.segment_map.segments),"nonempty_segments":sum(x.nonempty for x in item.segment_map.segments),"geometry":_geometry(config)})
    _atomic_json(config.run_dir/"status.json",{"schema_version":SCHEMA_VERSION,"status":"prepared","prepared_at_utc":_utc(),"seed":config.seed,"commit":_commit(),"geometry":_geometry(config)}); _append_jsonl(config.run_dir/"progress.jsonl",{"event":"prepared","at_utc":_utc()}); return segment_path


# Checkpoints and inference ----------------------------------------------------------

def _device(c:E4Config)->torch.device: return torch.device("cuda" if c.device=="auto" and torch.cuda.is_available() else ("cpu" if c.device=="auto" else c.device))
def _seed(seed:int)->None: random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def _atomic_checkpoint(path:Path,state:dict[str,Any])->None: temp=path.with_name(path.name+".tmp"); torch.save(state,temp); os.replace(temp,path)
def _checkpoint(c:E4Config,epoch:int,g:ConditionalGenerator,d:PatchDiscriminator,go:torch.optim.Optimizer,do:torch.optim.Optimizer,best:float)->dict[str,Any]: return {"schema_version":SCHEMA_VERSION,"epoch":epoch,"best_score":best,"geometry":_geometry(c),"composer_to_index":{x:i for i,x in enumerate(COMPOSERS)},"generator":g.state_dict(),"discriminator":d.state_dict(),"generator_optimizer":go.state_dict(),"discriminator_optimizer":do.state_dict(),"seed":c.seed}
def _load_models(c:E4Config,checkpoint:Path,device:torch.device)->ConditionalGenerator:
    payload=torch.load(checkpoint,map_location=device,weights_only=False)
    if payload.get("schema_version")!=SCHEMA_VERSION or payload.get("geometry")!=_geometry(c) or payload.get("composer_to_index")!={x:i for i,x in enumerate(COMPOSERS)}: raise ValueError("checkpoint geometry or artist mapping does not match configuration")
    model=ConditionalGenerator(conv_dim=c.conv_dim,residual_blocks=c.residual_blocks).to(device); model.load_state_dict(payload["generator"]); model.eval(); return model
def _infer_piece(model:ConditionalGenerator,piece:PieceSegments,target:int,device:torch.device)->tuple[Any,int]:
    decoded=[]; fallback=0
    with torch.no_grad():
        for seg in piece.segment_map.segments:
            x=torch.from_numpy(seg.data.astype(np.float32))[None].to(device); y=(torch.sigmoid(model(x,torch.tensor([target],device=device)))>=.5).to(torch.uint8)[0].cpu().numpy(); y[:,~seg.mask.astype(bool),:]=0
            if seg.nonempty and not y[:,seg.mask.astype(bool),:].any(): y=seg.data.copy(); fallback+=1
            decoded.append(Segment(seg.index,seg.start_bar,seg.bars,y,seg.mask.copy()))
    return stitch_segments(piece.segment_map,decoded),fallback


# Training and public stages ---------------------------------------------------------

def _run_training(c:E4Config,*,smoke:bool)->Path:
    assert c.run_dir
    _,parts,rows=_inputs(c)
    # Smoke deliberately avoids materialising the full corpus: it is a harness
    # gate, not a shortened pilot.  ``prepare`` remains the explicit full-index
    # stage used before a real training run.
    if smoke:
        smoke_parts={"train":set(),"validation":set(),"test":set()}
        for composer in COMPOSERS:
            candidates=sorted(sample for sample in parts["train"] if rows[sample]["composer"]==composer)
            smoke_parts["train"].update(candidates[:c.smoke_per_composer])
        # Six full outputs are also domain-balanced (normally two per artist).
        per_artist = max(1, c.smoke_files // len(COMPOSERS))
        for composer in COMPOSERS:
            candidates = sorted(sample for sample in parts["validation"] if rows[sample]["composer"] == composer)
            smoke_parts["validation"].update(candidates[:per_artist])
        if len(smoke_parts["validation"]) < c.smoke_files:
            smoke_parts["validation"].update(sorted(parts["validation"] - smoke_parts["validation"])[:c.smoke_files - len(smoke_parts["validation"])])
        parts=smoke_parts
    elif not (c.run_dir/"segments.jsonl").is_file():
        prepare_e4(c)
    collections=_pieces(c,parts,rows); datasets={split:SegmentDataset(value,split=split) for split,value in collections.items()}; assert_split_isolation(datasets); train:SegmentDataset|Subset=datasets["train"]
    if smoke:
        selected=[]
        for label in range(3): selected.extend([i for i,e in enumerate(datasets["train"].index) if datasets["train"].composer_to_index[e.composer]==label][:c.smoke_per_composer])
        train=Subset(datasets["train"],selected)
    _seed(c.seed); device=_device(c); g=ConditionalGenerator(conv_dim=c.conv_dim,residual_blocks=c.residual_blocks).to(device); d=PatchDiscriminator(conv_dim=c.conv_dim).to(device); go=torch.optim.Adam(g.parameters(),lr=.0002,betas=(.5,.999)); do=torch.optim.Adam(d.parameters(),lr=.0002,betas=(.5,.999)); start=0; best=float("-inf"); last=c.run_dir/"last.pt"
    if not smoke and last.is_file():
        state=torch.load(last,map_location=device,weights_only=False); _load_models(c,last,device); g.load_state_dict(state["generator"]); d.load_state_dict(state["discriminator"]); go.load_state_dict(state["generator_optimizer"]); do.load_state_dict(state["discriminator_optimizer"]); start,best=int(state["epoch"]),float(state["best_score"])
    sampler=BalancedComposerSampler(datasets["train"],seed=c.seed) if not smoke else None; loader=DataLoader(train if smoke else datasets["train"],batch_size=c.batch_size,sampler=sampler,shuffle=False if sampler else True); epochs=c.smoke_epochs if smoke else c.max_epochs; _atomic_json(c.run_dir/"status.json",{"schema_version":SCHEMA_VERSION,"status":"smoke_running" if smoke else "training","started_at_utc":_utc()})
    for epoch in range(start,epochs):
        if sampler: sampler.set_epoch(epoch)
        values=[]
        for raw in loader:
            batch={k:v.to(device) for k,v in raw.items() if k in {"x","mask","source"}}; values.append(gan_step(g,d,go,do,batch).generator_loss)
        score=-float(np.mean(values)); _atomic_checkpoint(last,_checkpoint(c,epoch+1,g,d,go,do,max(best,score)))
        if score>best: best=score; _atomic_checkpoint(c.run_dir/"best.pt",_checkpoint(c,epoch+1,g,d,go,do,best))
        _append_jsonl(c.run_dir/"progress.jsonl",{"event":"epoch_completed","stage":"smoke" if smoke else "train","epoch":epoch+1,"generator_loss":-score,"at_utc":_utc()})
    if smoke:
        model=_load_models(c,c.run_dir/"best.pt",device); selected=collections["validation"][:c.smoke_files]; outputs=c.run_dir/"smoke_midi"; results=[]
        for idx,piece in enumerate(selected):
            target=(COMPOSERS.index(piece.composer)+1)%3; rendered,fallback=_infer_piece(model,piece,target,device); path=outputs/f"{idx:02d}_{piece.sample_id}_to_{COMPOSERS[target]}.mid"; MidiPrettyPrinter().write(rendered,path); parsed=MidiParser().parse(path); results.append({"sample_id":piece.sample_id,"path":str(path),"parsed":True,"end_tick":piece.segment_map.end_tick,"result_end_tick":max((n.tick+n.duration_ticks for n in parsed.notes),default=0),"fallback_segments":fallback})
        passed=len(results)==c.smoke_files and all(x["result_end_tick"]==x["end_tick"] for x in results); _atomic_json(c.run_dir/"smoke.json",{"files":results,"passed":passed,"epochs":c.smoke_epochs}); status="smoke_passed" if passed else "smoke_failed"
    else: status="training_completed"
    _atomic_json(c.run_dir/"status.json",{"schema_version":SCHEMA_VERSION,"status":status,"finished_at_utc":_utc(),"best_score":best}); return c.run_dir/("smoke.json" if smoke else "best.pt")
def smoke_e4(c:E4Config)->Path: return _run_training(c,smoke=True)
def train_e4(c:E4Config)->Path: return _run_training(c,smoke=False)
def evaluate_e4(c:E4Config)->Path:
    assert c.run_dir; checkpoint=c.run_dir/"best.pt"
    if not checkpoint.is_file(): raise ValueError("best.pt is missing; run train or smoke first")
    _,parts,rows=_inputs(c); pieces=_pieces(c,parts,rows); device=_device(c); model=_load_models(c,checkpoint,device); results=[]
    for piece in pieces["validation"]:
        for target in range(3):
            if target!=COMPOSERS.index(piece.composer): output,fallback=_infer_piece(model,piece,target,device); results.append({"sample_id":piece.sample_id,"source":piece.composer,"target":COMPOSERS[target],"end_tick":piece.segment_map.end_tick,"result_end_tick":max((n.tick+n.duration_ticks for n in output.notes),default=0),"fallback_segments":fallback})
    path=c.run_dir/"evaluation.json"; _atomic_json(path,{"scope":"inner_validation_only","results":results}); return path
def report_e4(c:E4Config)->Path:
    assert c.run_dir; status=json.loads((c.run_dir/"status.json").read_text(encoding="utf-8")); smoke=c.run_dir/"smoke.json"; result=json.loads(smoke.read_text(encoding="utf-8")) if smoke.is_file() else None; text="# E4 report\n\nStatus: `"+str(status.get("status"))+"`.\n\n"+(f"Smoke gate: `{'PASS' if result['passed'] else 'FAIL'}` for {len(result['files'])} files.\n" if result else "No smoke result recorded.\n"); path=c.run_dir/"report.md"; path.write_text(text,encoding="utf-8"); return path
