#!/usr/bin/env python3
"""Verify a deliberately redacted historical export, not its original full bundle.
No training, tuning, network requests, or modification of retained evidence.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, json, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import numpy as np
from replaystudy import evidence as e, stages, data, runner

def read(p): return json.loads(p.read_text())
def require(ok,message):
    if not ok: raise ValueError(message)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_inventory():
    inventory=read(ROOT/'PUBLIC-FILES.json')
    transport=read(ROOT/'TRANSPORT.json') if (ROOT/'TRANSPORT.json').exists() else {'parts':[]}
    transport_paths={'TRANSPORT.json'} | {v['path'] for v in transport['parts']}
    require(all(Path(v).name==v and v.startswith('evidence.tar.gz.part-') for v in transport_paths-{'TRANSPORT.json'}),'Invalid transport path')
    actual=set()
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT).as_posix()
        if rel=='PUBLIC-FILES.json' or rel in transport_paths:continue
        if '__pycache__' in p.parts and p.suffix=='.pyc':continue
        actual.add(rel)
    require(actual==set(inventory),'Public file allowlist differs (missing or unexpected file)')
    for name,digest in inventory.items():
        p=ROOT/name
        require(not p.is_symlink() and sha(p)==digest,'Public hash mismatch: '+name)
    return len(inventory)
def verify_bundle_export(name,export):
    b=ROOT/'results'/name; manifest=read(b/'manifest.json'); inventory=read(b/'artifacts.json')
    require(sha(b/'manifest.json')==export['original_manifest_sha256'],'Original manifest changed')
    require(sha(b/'artifacts.json')==export['original_inventory_sha256'],'Original inventory changed')
    omitted=export['omitted_provenance']
    require(all(k.startswith('provenance/') and inventory.get(k)==v for k,v in omitted.items()),'Invalid omission catalog')
    actual=e._artifact_inventory(b)
    require(actual=={k:v for k,v in inventory.items() if k not in omitted},'Retained scientific artifact mismatch')
    require(not any((b/k).exists() for k in ('FAILED','INTERRUPTED','RUNNING')),'Unclean lifecycle')
    complete=read(b/'COMPLETE')
    require(complete['artifacts_sha256']==sha(b/'artifacts.json'),'Original completion hash mismatch')
    expected=e._validate_expected(manifest['stage'],manifest['expected_cases']); schema=manifest['required_fields']
    e._validate_schema(manifest['stage'],schema)
    records=e._read_cases(b/'cases.jsonl');seen=set()
    for row in records:e._validate_record(b,row,expected,schema,seen,manifest['stage'])
    require(seen==set(expected) and complete['case_count']==len(expected),'Incomplete case coverage')
    if manifest['stage']!='software':e._validate_research_pairing(records)
    return records

def verify_metrics(rows,reproduce):
    b=ROOT/'results/evaluation-v2'; summary=read(ROOT/'results/evaluation-summary-v2.json'); diag=read(ROOT/'results/evaluation-diagnostics-v2.json')
    prepared=ROOT/'data/prepared-v1'; m=read(prepared/'manifest.json')
    for desc in m['arrays'].values():require(sha(prepared/desc['file'])==desc['sha256'],'Split/duplicate audit array changed')
    raw=ROOT/'data/test-assets'
    for name in ('t10k-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz'):
        require(sha(raw/name)==m['assets'][name]['sha256'],'Official test asset changed')
    labels=data._read_asset(raw,'t10k-labels-idx1-ubyte.gz')
    x=data.preprocess_images(data._read_asset(raw,'t10k-images-idx3-ubyte.gz')) if reproduce else None
    keep=np.load(prepared/'test_keep_ids.npy',allow_pickle=False); require(len(keep)==10000,'Unexpected duplicate filtering')
    by={};audit_count=checkpoint_count=0
    for row in rows:
        p=b/row['case_id'];pairs=np.asarray(row['class_pairs']);matrix=[]
        for cp in row['checkpoints']:
            pred=np.load(p/f'test-predictions-{cp["block"]}.npy',allow_pickle=False)
            require(pred.shape==(10000,) and pred.dtype==np.uint8 and pred.max()<10,'Invalid prediction')
            recalls=np.array([np.mean(pred[labels==k]==k) for k in range(10)])
            matrix.append([float(recalls[pair].mean()) for pair in pairs]);checkpoint_count+=1
            if reproduce:
                with np.load(b/cp['path'],allow_pickle=False) as z: params=tuple(z[f'p{i}'] for i in range(4))
                _,_,actual=runner.evaluate(params,x,labels,pairs)
                require(np.array_equal(actual,pred),'Checkpoint does not reproduce saved predictions')
        require(matrix==row['metrics']['test_matrix']==row['metrics']['full_test_matrix'],'Metric matrix mismatch')
        require(recalls.tolist()==row['metrics']['test_recall']==row['metrics']['full_test_recall'],'Class recall mismatch')
        mm=np.asarray(matrix); acq=np.diag(mm).mean(); forget=np.mean([max(mm[k:4,k])-mm[4,k] for k in range(4)])
        sr=next(v for v in summary['per_seed'][row['arm']] if v['seed']==row['seed'])
        for key,value in [('final_macro',recalls.mean()),('immediate_acquisition',acq),('forgetting',forget)]:require(abs(sr[key]-value)<1e-14,'Summary metric mismatch')
        audits=e._read_cases(p/'mechanism-audit.jsonl');audit_count+=len(audits)
        dr=next(v for v in diag['per_case'] if v['case_id']==row['case_id'])
        for key in ('candidate_rank_vs_exhaustive_harm','selected_mean_harm','eligible_mean_harm','mir_highloss_overlap','selected_minus_shuffled_old_loss'):
            values=[v[key] for v in audits if v[key] is not None]
            require(abs(dr['audit_mean_'+key]-statistics.mean(values))<1e-14,'Audit summary mismatch')
        frequency=np.zeros(60000,np.uint32);coverage=[]
        for t in e._read_cases(p/'training/selection-trace.jsonl'):
            coverage.append(t['coverage'])
            if t['historical']:
                for key in ('selected_ids','extra_selected_ids'):np.add.at(frequency,np.asarray(t[key],np.int64),1)
        require(np.array_equal(frequency,np.load(p/'selection-frequency.npy',allow_pickle=False)),'Selection frequency mismatch')
        require(np.array_equal(coverage,np.load(p/'memory-class-coverage.npy',allow_pickle=False)),'Coverage mismatch')
        require(frequency.sum()==row['ledger']['historical_presentations']+row['ledger']['extra_historical_presentations'],'Presentation ledger mismatch')
        by[row['seed'],row['arm']]=float(recalls.mean())
    for arm in ('highloss','uniform','shuffledscore','uniform-extra-replay'):
        differences=np.array([by[s,'mir']-by[s,arm] for s in range(2001,2021)])
        random=np.random.Generator(np.random.PCG64(73103));means=differences[random.integers(0,20,(10000,20))].mean(1)
        ci=np.percentile(means,[2.5,97.5]);avg=differences.mean();saved=summary['comparisons'][arm]
        decision='supports_prespecified_threshold' if avg>=.01 and ci[0]>0 else 'practical_threshold_unsupported' if ci[1]<.01 else 'inconclusive'
        require(np.array_equal(differences,saved['individual_differences']) and abs(avg-saved['mean'])<1e-14 and np.allclose(ci,saved['ci95'],rtol=0,atol=1e-14) and decision==saved['decision'],'Bootstrap comparison mismatch')
    require((len(rows),checkpoint_count,audit_count)==(100,500,8000),'Evaluation counts mismatch')
    return {'evaluation_cases':len(rows),'checkpoint_prediction_pairs':checkpoint_count,'audits':audit_count,'all_predictions_reproduced':reproduce,'primary_decision':summary['comparisons']['highloss']['decision']}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reproduce-predictions',action='store_true');args=parser.parse_args()
    count=verify_inventory(); exports=read(ROOT/'bundle-exports.json');rows={name:verify_bundle_export(name,v) for name,v in exports.items()}
    freeze=read(ROOT/'source-freeze-original.json')
    for name,digest in freeze['files'].items():
        if name.endswith('.py'):require(sha(ROOT/'source'/name)==digest,'Frozen Python changed: '+name)
    lock=read(ROOT/'results/evaluation-lock-public.json')
    require(lock['rates']==stages.select_rates(rows['development-v2']),'Learning-rate selection mismatch')
    require(lock['data_manifest_sha256']==sha(ROOT/'data/prepared-v1/manifest.json'),'Data binding mismatch')
    require(lock['source_freeze_sha256']==sha(ROOT/'source-freeze-original.json'),'Source binding mismatch')
    result=verify_metrics(rows['evaluation-v2'],args.reproduce_predictions)
    print(json.dumps({'public_files':count,'development_cases':len(rows['development-v2']),'historical_full_bundle_verification':False,'public_export_verification':True,**result},indent=2))
if __name__=='__main__':main()
