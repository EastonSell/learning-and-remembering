"""Auditable staged entry points. Imports NumPy only after launcher thread gate."""
import argparse, hashlib, json, os, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from . import core as c, runner as r
ROOT=Path(__file__).resolve().parents[1]

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def source_files():
    return sorted(p for pattern in ['replaystudy/*.py','tests/*.py','docs/*.md','README.md','run.py'] for p in ROOT.glob(pattern))

def freeze(path):
    payload={'kind':'source-protocol-freeze-v1','created_at':now(),'python':platform.python_version(),'numpy':np.__version__,
      'files':{str(p.relative_to(ROOT)):r.file_sha(p) for p in source_files()},'architecture':[196,32,10],
      'arms':list(c.ARMS),'seeds_development':[1001,1002],'seeds_evaluation':list(range(2001,2021)),
      'rates':[.03,.10,.30],'parameter_count':6634,'budget_convention':'logical-scalar-kernels-v1'}
    r.dump(path,payload); return payload

def verify_freeze(path):
    v=read(path)
    if v.get('kind')!='source-protocol-freeze-v1' or not v.get('files'): raise ValueError('invalid source freeze')
    actual={str(p.relative_to(ROOT)):r.file_sha(p) for p in source_files()}
    if actual!=v['files']: raise ValueError('source/protocol/tests differ from frozen version; preserve and amend')
    if v['numpy']!=np.__version__ or v['python']!=platform.python_version(): raise ValueError('runtime version differs')
    return v

def authorize(freeze_path,authorization_path,stage):
    verify_freeze(freeze_path); a=read(authorization_path)
    if a.get('source_freeze_sha256')!=r.file_sha(freeze_path) or a.get('parent_go_ahead') is not True or stage not in a.get('approved_stages',[]): raise ValueError('missing explicit parent go-ahead for frozen stage')
    if not a.get('independent_review',{}).get('passed') or not a['independent_review'].get('reviewer') or not a['independent_review'].get('report_sha256'): raise ValueError('independent source/protocol review required')
    report=Path(a['independent_review']['report_path'])
    if r.file_sha(report)!=a['independent_review']['report_sha256']: raise ValueError('review report mismatch')
    return a

def select_rates(rows):
    keys=[(a['seed'],a['arm'],a['eta']) for a in rows]
    expected={(seed,arm,eta) for seed in (1001,1002) for arm in c.ARMS for eta in (.03,.1,.3)}
    if len(keys)!=30 or set(keys)!=expected: raise ValueError('complete unique development grid required')
    def score(rate,extra):
        values=[np.mean(a['metrics']['validation_recall']) for a in rows if a['eta']==rate and (a['arm']=='uniform-extra-replay')==extra]
        if not np.isfinite(values).all(): raise ValueError('nonfinite development results')
        return np.mean(values)
    return {'diagnostic_eta':min((.03,.1,.3),key=lambda eta:(-score(eta,False),eta)),
            'extra_eta':min((.03,.1,.3),key=lambda eta:(-score(eta,True),eta))}

def verify_development_binding(path,freeze_sha256,data_sha256):
    from .evidence import verify_bundle
    if verify_bundle(path)['stage']!='development': raise ValueError('lock requires a completed development-stage bundle')
    provenance=read(Path(path)/'manifest.json')['provenance']
    for key,wanted in [('freeze',freeze_sha256),('data',data_sha256)]:
        entries=provenance.get(key,[])
        if not entries or {entry['sha256'] for entry in entries}!={wanted}: raise ValueError('development '+key+' provenance differs from lock inputs')

def verify_lock(path):
    v=read(path)
    if v.get('kind')!='evaluation-lock-v1' or v.get('complete') is not True or v.get('no_prior_test_selection') is not True: raise ValueError('complete provenance-attested lock required')
    verify_freeze(v['freeze_path'])
    if r.file_sha(v['freeze_path'])!=v['source_freeze_sha256']: raise ValueError('freeze mismatch')
    if r.file_sha(v['data_manifest_path'])!=v['data_manifest_sha256']: raise ValueError('data manifest mismatch')
    verify_development_binding(v['development_path'],v['source_freeze_sha256'],v['data_manifest_sha256'])
    if r.file_sha(v['forecast_path'])!=v['forecast_sha256'] or verify_forecast(v['forecast_path'],v['freeze_path'])!=v['forecast']: raise ValueError('locked feasibility forecast differs')
    if r.file_sha(Path(v['development_path'])/'cases.jsonl')!=v['development_cases_sha256']: raise ValueError('development case mismatch')
    if v['seeds']!=list(range(2001,2021)) or v['architecture']!=[196,32,10] or v['arms']!=list(c.ARMS): raise ValueError('lock configuration changed')
    expected=select_rates(load_rows(v['development_path']))
    if v['rates']!=expected: raise ValueError('rates differ from fixed rule')
    r.check_forecast(v['forecast']); return v

def load_rows(path): return [json.loads(line) for line in (Path(path)/'cases.jsonl').read_text().splitlines() if line.strip()]

def make_lock(args):
    a=authorize(args.freeze,args.authorization,'lock')
    verify_development_binding(args.development,r.file_sha(args.freeze),r.file_sha(Path(args.prepared)/'manifest.json'))
    if a.get('no_prior_test_selection') is not True or not a.get('test_provenance_basis'): raise ValueError('explicit test provenance attestation and basis required')
    forecast=verify_forecast(args.forecast,args.freeze)
    manifest=Path(args.prepared)/'manifest.json'
    v={'kind':'evaluation-lock-v1','complete':True,'created_at':now(),'no_prior_test_selection':True,'test_provenance_basis':a['test_provenance_basis'],
       'freeze_path':str(Path(args.freeze).resolve()),'source_freeze_sha256':r.file_sha(args.freeze),
       'data_manifest_path':str(manifest.resolve()),'data_manifest_sha256':r.file_sha(manifest),
       'development_path':str(Path(args.development).resolve()),'development_cases_sha256':r.file_sha(Path(args.development)/'cases.jsonl'),
       'rates':select_rates(load_rows(args.development)),'architecture':[196,32,10],'arms':list(c.ARMS),'seeds':list(range(2001,2021)),
       'forecast':forecast,'forecast_path':str(Path(args.forecast).resolve()),'forecast_sha256':r.file_sha(args.forecast),'bootstrap':{'seed':73103,'resamples':10000},'primary':'mir-minus-highloss final macro accuracy',
       'practical_threshold':.01,'acquisition_tradeoff_threshold':.02,'amendments':[str(p.relative_to(ROOT)) for p in ROOT.glob('docs/*amendment*.md')]}
    r.dump(args.output,v); return v

def verify_forecast(path,freeze_path):
    from .evidence import verify_bundle
    p=Path(path); verify_bundle(p.parent)
    forecast=read(p); r.check_forecast(forecast)
    if forecast.get('synthetic_only') is not True or forecast.get('feasibility_wall_seconds',901)>900: raise ValueError('verified synthetic feasibility required')
    manifest=read(p.parent/'manifest.json')
    if manifest.get('stage')!='software': raise ValueError('feasibility must be software stage')
    if r.file_sha(freeze_path) not in [a['sha256'] for a in manifest['provenance'].get('freeze',[])]: raise ValueError('feasibility source freeze mismatch')
    return forecast

def disk_bytes(paths):
    files={p.resolve() for root in {Path(v).resolve() for v in paths} if root.exists() for p in root.rglob('*') if p.is_file()}
    return sum(p.stat().st_size for p in files)

def resource_paths(args,prior):
    return [Path(args.data_root),Path(args.prepared),Path(args.output),Path(args.forecast).parent]+[Path(v['output']) for v in prior if 'output' in v]

def check_resources(paths,*,prior_cpu,start_cpu,observations=None):
    disk=disk_bytes(paths); memory=r.rss(); cpu=prior_cpu+time.process_time()-start_cpu
    if observations is not None:
      for key,value in [('sampled_peak_output_bytes',disk),('sampled_peak_rss_bytes',memory),('sampled_total_cpu_seconds',cpu)]: observations[key]=max(observations.get(key,0),value)
    if disk>250*1024**2: raise ValueError('retained data/artifacts exceed 250 MiB')
    if memory>1024**3: raise MemoryError('sampled RSS exceeds 1 GiB')
    if cpu>=7200: raise TimeoutError('total CPU budget exhausted')

def training_provenance(args,stage):
    d={'source':[ROOT/'replaystudy',ROOT/'run.py',ROOT/'README.md'],'tests':ROOT/'tests','protocol':ROOT/'docs','data':Path(args.prepared)/'manifest.json','indices':list(Path(args.prepared).glob('*ids.npy')),'freeze':Path(args.freeze),'authorization':Path(args.authorization)}
    if stage=='evaluation': d['lock']=Path(args.lock)
    return d

def trace_analysis(path):
    frequency=np.zeros(60000,np.uint32); coverage=[]
    with open(path) as f:
      for line in f:
        row=json.loads(line); coverage.append(row['coverage'])
        if row['historical']:
          for key in ('selected_ids','extra_selected_ids'): np.add.at(frequency,np.asarray(row[key],np.int64),1)
    return frequency,np.asarray(coverage,np.uint16)

def run_stage(args,stage):
    from . import data
    from .evidence import Bundle,expected_cases,make_case_id
    bundle_cpu=time.process_time(); bundle_wall=time.perf_counter()
    r.require_threads()
    authorize(args.freeze,args.authorization,stage)
    forecast=verify_forecast(args.forecast,args.freeze)
    lock=verify_lock(args.lock) if stage=='evaluation' else None
    if lock is not None:
      if r.file_sha(args.freeze)!=lock['source_freeze_sha256'] or r.file_sha(Path(args.prepared)/'manifest.json')!=lock['data_manifest_sha256'] or r.file_sha(args.forecast)!=lock['forecast_sha256']: raise ValueError('evaluation inputs differ from locked freeze/data/forecast')
    cases=expected_cases(stage,**(lock['rates'] if lock else {}))
    # Development/evaluation share one append-only budget journal outside bundles.
    journal=Path(args.budget_journal)
    prior_cpu=0.; prior=[]
    if journal.exists():
      prior=[json.loads(v) for v in journal.read_text().splitlines()]
      if any(v.get('status')=='started' and not any(w.get('event')==v['event'] and w.get('status')=='finished' for w in prior) for v in prior): raise ValueError('unclosed prior budget reservation; reconcile interruption before proceeding')
      prior_cpu=sum(v.get('cpu_seconds',0) for v in prior if v.get('status')=='finished')
    paths=resource_paths(args,prior)
    stage_resource_peaks={}
    check=lambda: check_resources(paths,prior_cpu=prior_cpu,start_cpu=bundle_cpu,observations=stage_resource_peaks)
    event=now(); journal.parent.mkdir(parents=True,exist_ok=True)
    def journal_write(v):
      with open(journal,'a') as f: f.write(json.dumps(v,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
    journal_write({'event':event,'status':'started','stage':stage,'output':str(args.output)})
    try:
      with Bundle(args.output,stage=stage,expected_cases=cases,provenance=training_provenance(args,stage)) as bundle:
        check()
        values=data.load_development(args.data_root,args.prepared)
        check()
        # Held-out arrays stay in coordinator, never learner input.
        vx,vy=values['x_val'],values['y_val']
        audit_idx=np.concatenate([np.flatnonzero(vy==k)[:10] for k in range(10)])
        for case in cases:
          check()
          case_id=case['case_id']; path=bundle.path/case_id; path.mkdir(exist_ok=False)
          try:
            order,pairs=c.stream_order(values['y_train'],case['seed'])
            # IDs are raw official-training source indices, never local row indices.
            np.save(path/'stream-indices.npy',values['train_ids'][order],allow_pickle=False)
            r.dump(path/'class-pairs.json',pairs.tolist())
            audit_file=open(path/'mechanism-audit.jsonl','x')
            def sink(p,fx,fy,bx,by,bids,cand,selected,u):
              old=pairs[:(u-1)//500].reshape(-1)
              row=r.audit(p,fx,fy,bx,by,bids,cand,selected,case['eta'],vx[audit_idx],vy[audit_idx],old,case['seed'],u)
              audit_file.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n'); audit_file.flush(); os.fsync(audit_file.fileno())
            (path/'training').mkdir()
            try:
              result=r.train_stream(values['x_train'][order],values['y_train'][order],values['train_ids'][order],seed=case['seed'],arm=case['arm'],eta=case['eta'],output=path/'training',audit_sink=sink,
                cpu_deadline=time.process_time()+7200-prior_cpu-(time.process_time()-bundle_cpu))
            finally: audit_file.close()
            # Checkpoints retained before validation or official scoring.
            result['stream_sha256']=r.file_sha(path/'stream-indices.npy')
            result['class_pairs']=pairs.tolist()
            for cp in result['checkpoints']: cp['path']=str((path/'training'/cp['path']).relative_to(bundle.path))
            result.update(case,status='ok')
            frequency,coverage=trace_analysis(path/'training'/'selection-trace.jsonl')
            np.save(path/'selection-frequency.npy',frequency,allow_pickle=False); np.save(path/'memory-class-coverage.npy',coverage,allow_pickle=False)
            # Official data are not loaded until ALL locked evaluation training finishes.
            if stage=='development':
              matrix=[]; rec=None; scoring_start=time.perf_counter(); scoring_cpu_start=time.process_time()
              for cp in result['checkpoints']:
                with np.load(bundle.path/cp['path'],allow_pickle=False) as z: p=tuple(z[f'p{i}'] for i in range(4))
                rec,row,_=r.evaluate(p,vx,vy,pairs); matrix.append(row.tolist()); check()
              result['metrics']={'validation_matrix':matrix,'validation_recall':rec.tolist()}; result['ledger']['scoring_wall_seconds']=time.perf_counter()-scoring_start; result['ledger']['scoring_cpu_seconds']=time.process_time()-scoring_cpu_start; result['ledger']['scoring_operations']=5*(c.forward_ops(len(vy))+9*len(vy))
              bundle.append_case(result)
            else:
              r.dump(path/'trained-case.json',result)
            check()
          except BaseException as error:
            r.dump(path/'case-failure.json',{'case':case,'error_type':type(error).__name__,'error':str(error)})
            try: bundle.append_case({**case,'status':'failed','error_type':type(error).__name__,'error':str(error)})
            except Exception: pass # append-first validation may already have marked failure
            raise
        if stage=='evaluation':
          # No test predictions have occurred before this timestamp.
          check()
          test=data.load_test_for_scoring(args.data_root,args.prepared,lock_path=args.lock,verify_lock=verify_lock)
          check()
          r.dump(bundle.path/'first-test-scoring.json',{'timestamp':now(),'lock_sha256':r.file_sha(args.lock),'all_training_complete':True})
          keep=test['keep_ids']
          for case in cases:
            path=bundle.path/case['case_id']; result=read(path/'trained-case.json'); pairs=np.asarray(result['class_pairs'])
            matrix=[]; full=[]; rec=None; frec=None; scoring_start=time.perf_counter(); scoring_cpu_start=time.process_time()
            for cp in result['checkpoints']:
              with np.load(bundle.path/cp['path'],allow_pickle=False) as z: p=tuple(z[f'p{i}'] for i in range(4))
              frec,frow,pred=r.evaluate(p,test['x_test'],test['y_test'],pairs)
              rec=r.recalls(pred[keep],test['y_test'][keep]); row=np.array([rec[pair].mean() for pair in pairs])
              matrix.append(row.tolist()); full.append(frow.tolist())
              # Prediction hashes allow deterministic reproduction without bulky logits.
              np.save(path/f'test-predictions-{cp["block"]}.npy',pred.astype(np.uint8),allow_pickle=False)
              check()
            result['metrics']={'test_matrix':matrix,'full_test_matrix':full,'test_recall':rec.tolist(),'full_test_recall':frec.tolist()}
            result['ledger']['scoring_wall_seconds']=time.perf_counter()-scoring_start; result['ledger']['scoring_cpu_seconds']=time.process_time()-scoring_cpu_start; result['ledger']['scoring_operations']=5*(c.forward_ops(len(test['y_test']))+9*len(test['y_test']))
            bundle.append_case(result)
            check()
        check()
        bundle.complete()
        check()
    finally:
      journal_write({'event':event,'status':'finished','cpu_seconds':time.process_time()-bundle_cpu,'wall_seconds':time.perf_counter()-bundle_wall,'resource_samples_through_budget_close':stage_resource_peaks})
      try: check()
      except BaseException as error:
        marker=Path(args.output)/'FAILED'
        if marker.parent.is_dir() and not marker.exists(): r.dump(marker,{'exception_type':type(error).__name__,'message':str(error),'phase':'final resource audit'})
        raise

def summary(args):
    from .evidence import verify_bundle
    if verify_bundle(args.bundle)['stage']!='evaluation': raise ValueError('summary requires evaluation stage')
    rows=load_rows(args.bundle)
    if len(rows)!=100: raise ValueError('evaluation bundle required')
    indexed={(v['seed'],v['arm']):v for v in rows}; out={}
    for arm in c.ARMS:
      metrics=[]
      for seed in range(2001,2021):
        row=indexed[seed,arm]; matrix=np.asarray(row['metrics']['test_matrix']); rec=np.asarray(row['metrics']['test_recall'])
        metrics.append({'seed':seed,'final_macro':float(rec.mean()),'immediate_acquisition':float(np.diag(matrix).mean()),'forgetting':r.mean_forgetting(matrix),'recall':rec.tolist()})
      out[arm]=metrics
    comparisons={arm:r.paired_summary([out['mir'][i]['final_macro']-out[arm][i]['final_macro'] for i in range(20)]) for arm in c.ARMS if arm!='mir'}
    plasticity=np.mean([v['immediate_acquisition'] for v in out['mir']])-np.mean([v['immediate_acquisition'] for v in out['highloss']])
    result={'per_seed':out,'comparisons':comparisons,'primary':'highloss','secondary_intervals_descriptive':True,'plasticity_tradeoff':bool(plasticity<-.02),'mir_minus_highloss_acquisition':float(plasticity),'interpretation':'Algorithmic selector intervention only; no biological implementation or novel algorithm claim.'}
    r.dump(args.output,result)

def forecast_resources(rows,data_bytes,scoring_cpu):
    # 130 runs x 2500 base updates; two-fold CPU headroom covers audit/I/O.
    train_cpu=sum(row['ledger']['cpu_seconds']/row['ledger']['committed_base_updates'] for row in rows)*26*2500
    # Each arm has 6 development + 20 evaluation cases. Five test/validation scores.
    score_cpu=scoring_cpu*(30*5*5000+100*5*10000)/1000
    # Conservative linear trace extrapolation overcounts fixed checkpoints.
    output=sum((row['artifact_bytes']-row.get('checkpoint_bytes',0))/row['ledger']['committed_base_updates']*2500+row.get('checkpoint_bytes',0) for row in rows)*26
    return {'cpu_seconds':2*(train_cpu+score_cpu),'output_bytes':int(data_bytes+output+40*1024**2),
      'rss_bytes':int(max(row['ledger']['sampled_peak_rss_bytes'] for row in rows)+200*1024**2),
      'method':'2x synthetic measured CPU extrapolation; trace/checkpoint extrapolation plus 40 MiB audits/provenance; data included',
      'synthetic_only':True,'scoring_probe_examples':1000,'research_base_updates':325000}

def feasibility(args):
    from .evidence import Bundle,make_case_id
    started=time.perf_counter()
    r.require_threads()
    verify_freeze(args.freeze)
    output=Path(args.output)
    spec=output.parent/(output.name+'-synthetic-input.json')
    r.dump(spec,{'kind':'synthetic-software-only','seed':777,'samples':800,'features':196,'labels':'cyclic 0..9','no_official_data_predictions':True})
    cases=[{'case_id':make_case_id(777,arm,.1),'seed':777,'arm':arm,'eta':.1} for arm in c.ARMS]
    provenance={'source':[ROOT/'replaystudy',ROOT/'run.py',ROOT/'README.md'],'tests':ROOT/'tests','protocol':ROOT/'docs','data':spec,'indices':spec,'freeze':Path(args.freeze)}
    gen=np.random.default_rng(777); x=gen.random((800,196),dtype=np.float32); y=(np.arange(800)%10).astype(np.uint8)
    rows=[]
    with Bundle(output,stage='software',expected_cases=cases,provenance=provenance) as bundle:
      for case in cases:
        path=bundle.path/case['case_id']; path.mkdir()
        result=r.train_stream(x,y,np.arange(800,dtype=np.uint64),seed=777,arm=case['arm'],eta=.1,output=path,software_only=True,block_size=160)
        for cp in result['checkpoints']: cp['path']=str((path/cp['path']).relative_to(bundle.path))
        result.update(case,status='ok',metrics={'validation_recall':[0.]*10})
        result['checkpoint_bytes']=sum((bundle.path/cp['path']).stat().st_size for cp in result['checkpoints'])
        result['artifact_bytes']=sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
        bundle.append_case(result); rows.append(result)
        if time.perf_counter()-started>900: raise TimeoutError('15 minute feasibility gate')
      px=gen.random((1000,196),dtype=np.float32); py=(np.arange(1000)%10).astype(np.uint8)
      cpu=time.process_time(); r.evaluate(c.initialize(777),px,py,np.arange(10).reshape(5,2)); scoring_cpu=time.process_time()-cpu
      data_bytes=sum(p.stat().st_size for p in Path(args.data_root).rglob('*') if p.is_file())
      forecast=forecast_resources(rows,data_bytes,scoring_cpu); forecast['feasibility_wall_seconds']=time.perf_counter()-started
      r.dump(bundle.path/'forecast.json',forecast)
      r.check_forecast(forecast); bundle.complete()

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze'); f.add_argument('--output',required=True)
    v=sub.add_parser('verify-freeze'); v.add_argument('--freeze',required=True)
    q=sub.add_parser('feasibility')
    for a in ['freeze','output','data-root']: q.add_argument('--'+a,required=True)
    for cmd in ['development','evaluation','lock']:
      q=sub.add_parser(cmd)
      for a in ['freeze','authorization','prepared','forecast','output']: q.add_argument('--'+a,required=True)
      if cmd=='lock': q.add_argument('--development',required=True)
      else:
        q.add_argument('--data-root',required=True); q.add_argument('--budget-journal',required=True)
        if cmd=='evaluation': q.add_argument('--lock',required=True)
    q=sub.add_parser('summarize'); q.add_argument('--bundle',required=True); q.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='freeze': freeze(args.output)
    elif args.command=='verify-freeze': verify_freeze(args.freeze)
    elif args.command=='lock': make_lock(args)
    elif args.command=='feasibility': feasibility(args)
    elif args.command=='summarize': summary(args)
    else: run_stage(args,args.command)
if __name__=='__main__': main()
