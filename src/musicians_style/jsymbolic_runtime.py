"""Isolated Java invocation for one neutral MIDI; no classifier or scientific cohort."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import time

from .feature_backends.jsymbolic import parse_exports
from .feature_feasibility import sanitize_midi
from .provenance import sha256_file, write_json


class JSymbolicRuntime:
    def __init__(self, checkout: Path, distribution: Path, java: Path, *, timeout=120):
        self.checkout=checkout.resolve(); self.distribution=distribution.resolve()
        self.java=java.resolve(); self.timeout=timeout
        self.lock=json.loads((checkout/'requirements/research-ab-jsymbolic.lock.json').read_text(encoding='utf-8'))
        self.config=checkout/self.lock['config']
        self.expected=json.loads((checkout/self.lock['schema']).read_text(encoding='utf-8'))

    def verify(self):
        for relative,digest in self.lock['files'].items():
            path=self.distribution/relative
            if not path.is_file() or sha256_file(path)!=digest:
                raise ValueError('pinned jSymbolic distribution mismatch: '+relative)
        jars={p.relative_to(self.distribution).as_posix() for p in self.distribution.rglob('*.jar')}
        if jars != set(self.lock['files']):
            raise ValueError('unexpected jSymbolic library set')
        for name in ('config','schema'):
            if sha256_file(self.checkout/self.lock[name])!=self.lock[name+'_sha256']:
                raise ValueError('pinned jSymbolic '+name+' mismatch')
        process=subprocess.run([str(self.java),'-version'],capture_output=True,text=True,timeout=10,check=True)
        return dict(java_executable=str(self.java),java_version=(process.stdout+process.stderr).strip(),
                    java_sha256=sha256_file(self.java),distribution=str(self.distribution),lock=self.lock)

    def extract(self, source: Path, output: Path):
        output=output.resolve()
        if output.exists():
            raise ValueError('fresh extraction destination required')
        output.mkdir(parents=True)
        started=time.perf_counter()
        record={}
        try:
            record['runtime']=self.verify()
            record['input_sha256']=sha256_file(source)
            record['sanitation']=sanitize_midi(source,output/'input.mid')
            command=[str(self.java),*self.lock['java_options'],'-jar',str(self.distribution/'jSymbolic2.jar'),
                '-configrun',str(self.config.resolve()),str(output/'input.mid'),str(output/'values.xml'),str(output/'definitions.xml')]
            record['command']=command
            with (output/'java.log').open('w',encoding='utf-8') as log:
                process=subprocess.run(command,cwd=self.distribution,stdout=log,stderr=subprocess.STDOUT,
                    timeout=self.timeout,check=False)
            record['returncode']=process.returncode
            if process.returncode:
                raise RuntimeError('jSymbolic process exit '+str(process.returncode))
            result=parse_exports(output/'values.xml',output/'definitions.xml',self.expected)
            record.update(status='success',**result)
        except Exception as exc:
            record.update(status='failure',failure=dict(type=type(exc).__name__,message=str(exc)))
        record['extraction_seconds']=time.perf_counter()-started
        write_json(output/'result.json',record)
        return record
