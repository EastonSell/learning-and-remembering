"""Stdlib-only admission and terminal evidence for one finite diagnostic.

Not a supervisor: fatal hard kills can prevent an internal failure receipt.
Success additionally requires a separately observed zero process exit.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

OUTPUT_LIMIT=32*1024**2
FINAL_RESERVE=64*1024
FAILURE_RESERVE=8*1024
COMPLETE_RESERVE=8*1024


def _encode(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(65536),b''): h.update(chunk)
    return h.hexdigest()


class Admission:
    def __init__(self,output,started=None):
        self.output=Path(output); self.cpu0,self.wall0=started or (time.process_time(),time.monotonic())
        self.owned=False; self.limits_installed=False
    def install_limits(self):
        def stopped(signum,frame): raise RuntimeError(f'catchable resource/termination signal {signum}; bundle invalid')
        for sig in (signal.SIGXCPU,signal.SIGXFSZ,signal.SIGALRM,signal.SIGTERM): signal.signal(sig,stopped)
        # Respect any stricter inherited limits; never broaden them.
        for name,want in [('RLIMIT_CPU',(180,181)),('RLIMIT_AS',(768*1024**2,)*2),('RLIMIT_FSIZE',(OUTPUT_LIMIT,)*2),('RLIMIT_CORE',(0,0))]:
            which=getattr(resource,name); old=resource.getrlimit(which)
            hard=want[1] if old[1]==resource.RLIM_INFINITY else min(want[1],old[1])
            soft=min(want[0],hard) if old[0]==resource.RLIM_INFINITY else min(want[0],old[0],hard)
            resource.setrlimit(which,(soft,hard))
        signal.setitimer(signal.ITIMER_REAL,max(.001,300-(time.monotonic()-self.wall0)))
        self.limits_installed=True
    def reserve_output(self):
        self.output.mkdir(parents=False,exist_ok=False); self.owned=True
    def snapshot(self):
        return {'cpu':time.process_time()-self.cpu0,'wall':time.monotonic()-self.wall0,
            'rss_high_water_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'output':sum(p.stat().st_size for p in self.output.iterdir() if p.is_file()) if self.owned else 0}
    def check(self):
        v=self.snapshot()
        for key,cap in [('cpu',180),('wall',300),('rss_high_water_bytes',768*1024**2),('output',OUTPUT_LIMIT)]:
            if v[key]>cap: raise RuntimeError(f'observed {key} gate exceeded: {v[key]}>{cap}')
    def _write_new(self,name,encoded):
        if Path(name).name!=name: raise ValueError('only leaf evidence filenames permitted')
        with (self.output/name).open('xb') as f:
            f.write(encoded); f.flush(); os.fsync(f.fileno())
    def write_json(self,name,value,terminal=False):
        if (self.output/name).exists(): raise FileExistsError(self.output/name)
        encoded=_encode(value); reserve=(FAILURE_RESERVE+COMPLETE_RESERVE) if terminal else FINAL_RESERVE
        self.check()
        if self.snapshot()['output']+len(encoded)+reserve>OUTPUT_LIMIT: raise RuntimeError('reserved terminal-evidence output gate exceeded')
        self._write_new(name,encoded); self.check()
    def finalize_success(self,*,states,seed_clusters):
        self.check()
        files={p.name:{'bytes':p.stat().st_size,'sha256':_sha(p)} for p in sorted(self.output.iterdir()) if p.is_file()}
        if 'FAILURE.json' in files or 'COMPLETE.json' in files: raise ValueError('terminal artifact already exists')
        before=self.snapshot(); total=before['output']
        receipt={'status':'posthoc_diagnostic_complete','states':states,'seed_clusters':seed_clusters,'files':files,
            'artifact_bytes_including_complete':total,
            'resources':{'sampled_before_completion_write':before},
            'success_requires':'observed external exit 0, no FAILURE, valid receipt/file hashes and byte total; missing COMPLETE is never success'}
        # Its own byte count is self-referential only through this integer field.
        for _ in range(10):
            encoded=_encode(receipt); expected=total+len(encoded)
            if expected==receipt['artifact_bytes_including_complete']: break
            receipt['artifact_bytes_including_complete']=expected
        else: raise RuntimeError('completion byte-count fixed point failed')
        if expected+FAILURE_RESERVE>OUTPUT_LIMIT: raise RuntimeError('completion would consume reserved failure-evidence bytes')
        self._write_new('COMPLETE.json',encoded)
        self.check(); verify_completion(self.output,exit_code=0); self.check()
        # Do not disable timers. A later exception/nonzero exit invalidates success.
    def record_failure(self,exc):
        if not self.owned: return False  # Never alter a colliding/unowned output directory.
        try:
            value={'status':'entire_bundle_invalid','type':type(exc).__name__,'reason':str(exc)[:1024],
                'resources_sampled_during_failure':self.snapshot(),
                'receipt_limit':'best effort; hard kills, allocation or filesystem failures may prevent this receipt; preserve external exit/partial files'}
            encoded=_encode(value)
            if len(encoded)>FAILURE_RESERVE or self.snapshot()['output']+len(encoded)>OUTPUT_LIMIT: return False
            self._write_new('FAILURE.json',encoded); return True
        except BaseException: return False


def verify_completion(output,*,exit_code):
    output=Path(output)
    if exit_code!=0 or (output/'FAILURE.json').exists(): raise ValueError('nonzero external exit or failure receipt invalidates bundle')
    receipt=json.loads((output/'COMPLETE.json').read_text())
    files=receipt['files']; actual={p.name for p in output.iterdir() if p.is_file()}
    if actual!=set(files)|{'COMPLETE.json'}: raise ValueError('missing or unexpected evidence file')
    for name,info in files.items():
        if Path(name).name!=name or (output/name).stat().st_size!=info['bytes'] or _sha(output/name)!=info['sha256']: raise ValueError('evidence file mismatch')
    total=sum((output/name).stat().st_size for name in actual)
    if total!=receipt['artifact_bytes_including_complete'] or total+FAILURE_RESERVE>OUTPUT_LIMIT: raise ValueError('terminal byte total mismatch or reserve consumed')
    return receipt
