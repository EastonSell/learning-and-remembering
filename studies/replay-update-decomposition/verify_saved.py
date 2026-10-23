#!/usr/bin/env python3
"""Verify only packaged saved scalar JSON/CSV and file hashes; no model execution.

Requires installed NumPy 2.3.5 for the recorded fixed scalar bootstrap. Imports no
reference model, image decoder, checkpoint loader, training runner or network API.
Writes only stdout. Keep any redirected output outside this closed package.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import numpy as np

SEEDS=(2005,2006,2007,2008)
BLOCKS=(0,1,2,3)
CELLS=('00','01','10','11')
COMPARISONS=('mir_minus_highloss','mir_minus_uniform','random_tie_mir_minus_highloss','random_tie_contrast_shift')
COMPONENTS=('weight_at_original','location_at_deployed','interaction','total_sequential_minus_joint','weight_at_updated','location_at_unit','reconciliation_residual')
FREEZE='8c70ca1b3a7ea529bddadd360ee78a40be5115f1dc5650aed2305bee2e85876c'
COMPLETE='51853d31e6d7f50b64da1cacfbcda51a582564c6c4e0c0c4f5b0102c2146eca4'
SUMMARY='e411cec1236837fe6e8bbb9bed3e20615479430b00508b579cc7ba1f028a7e12'

class VerificationError(ValueError):pass

def require(ok,message):
    if not ok:raise VerificationError(message)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate JSON key')
            result[key]=value
        return result
    def invalid(value):raise VerificationError('nonfinite JSON number: '+value)
    try:result=json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=invalid)
    except (OSError,ValueError) as exc:raise VerificationError(str(exc)) from exc
    def finite(value):
        if isinstance(value,float):require(math.isfinite(value),'nonfinite JSON value')
        elif isinstance(value,dict):
            for item in value.values():finite(item)
        elif isinstance(value,list):
            for item in value:finite(item)
    finite(result);return result

def compare_summary(expected,actual):
    require(expected==actual,'saved summary arithmetic or descriptive bootstrap mismatch')

def inventory(root):
    manifest=load_json(root/'PUBLIC-FILES.json')
    expected=set(manifest['files'])|{'PUBLIC-FILES.json'}
    for name in expected:
        p=PurePosixPath(name)
        require(not p.is_absolute() and '..' not in p.parts and str(p)==name,'unsafe allowlist path')
    paths=list(root.rglob('*'))
    require(not any(p.is_symlink() for p in paths),'symlink in package')
    actual={str(p.relative_to(root)) for p in paths if p.is_file()}
    require(actual==expected,'closed allowlist inventory mismatch')
    expected_dirs={str(parent) for name in expected for parent in PurePosixPath(name).parents if str(parent)!='.'}
    require({str(p.relative_to(root)) for p in paths if p.is_dir()}==expected_dirs,'closed allowlist directory mismatch')
    for name,item in manifest['files'].items():
        p=root/name
        require(p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'allowlist hash/size mismatch: '+name)
    return manifest

def recompute_row(row):
    require((row['seed'],row['block']) in set(itertools.product(SEEDS,BLOCKS)),'unexpected state')
    require(row['eta']==.3 and row['rng_domain']==7310501,'state parameters changed')
    metrics={};families={}
    require(set(row['cells'])==set(CELLS),'cell inventory changed')
    candidates=dict(zip(row['candidate_ids'],row['candidate_labels']))
    require(len(candidates)==50,'candidate inventory mismatch')
    require(len(row['fresh_ids'])==16 and len(set(row['fresh_ids']))==16,'fresh inventory mismatch')
    require(not set(row['fresh_ids'])&set(row['resident_ids']),'fresh/resident overlap')
    for panel in ('old','new'):
        ids=row[panel+'_panel_ids'];labels=row[panel+'_panel_labels']
        require(len(set(ids))==len(ids) and len(ids)==len(labels),'panel IDs duplicated')
        require(all(labels.count(k)==500 for k in row[panel+'_labels']),'panel class counts')
        require(not set(ids)&(set(row['fresh_ids'])|set(row['resident_ids'])),'evaluator/training overlap')
    require(not set(row['old_panel_ids'])&set(row['new_panel_ids']),'old/new panel overlap')
    require(set(candidates)<=set(row['resident_ids']),'candidate membership mismatch')
    for name,selection in row['selections'].items():
        ids=selection['ids'];labels=selection['labels']
        require(len(ids)==10 and len(set(ids))==10 and all(i in candidates for i in ids),'selection membership/count')
        require(labels==[candidates[i] for i in ids],'selection labels changed')
        require({str(k):labels.count(int(k)) for k in row['quota']}==row['quota'],'selection quota changed')
    for name,scorefield in (('mir','signed_virtual_delta'),('highloss','current_loss')):
        scores=dict(zip(row['candidate_ids'],row[scorefield]));chosen=[]
        for label in sorted(map(int,row['quota'])):
            ids=[i for i in row['candidate_ids'] if candidates[i]==label]
            chosen+=sorted(ids,key=lambda i:(-scores[i],i))[:row['quota'][str(label)]]
        require(chosen==row['selections'][name]['ids'],'saved signed selector ranking mismatch')
    for cell in CELLS:
        branches=row['cells'][cell]['branches']
        require(set(branches)==set(row['selections']) and len(branches)==8,'branch inventory changed')
        weights={'fresh':16/26,'replay':10/26} if cell[0]=='0' else {'fresh':1.,'replay':1.}
        require(row['cells'][cell]['weights']==weights,'factor weights changed')
        location=row['state_sha256'] if cell[1]=='0' else row['fresh_step_sha256']['deployed' if cell[0]=='0' else 'unit']
        for panel in ('old','new'):
            loss=lambda name:branches[name][panel+'_loss']
            mh=loss('mir')-loss('highloss');rt=loss('mir_random_ties')-loss('highloss_random_ties')
            contrasts={'mir_minus_highloss':mh,'mir_minus_uniform':loss('mir')-float(np.mean([loss(f'uniform_{i}') for i in range(4)])),
                       'random_tie_mir_minus_highloss':rt,'random_tie_contrast_shift':rt-mh}
            require(contrasts==row['cells'][cell]['contrasts'][panel],'saved contrast mismatch')
            for comparison,value in contrasts.items():metrics[f'{panel}.{comparison}.cell{cell}']=value
        for selection,branch in branches.items():
            require(branch['start_state_sha256']==row['state_sha256'],'branch start mismatch')
            require(branch['replay_gradient_state_sha256']==location,'replay gradient location mismatch')
            for panel in ('old','new'):
                for control,baseline in row['baselines'].items():
                    require(branch[f'{panel}_change_from_{control}']==branch[panel+'_loss']-baseline[panel],'saved control delta mismatch')
            for field,value in branch.items():
                if field.startswith(('old_','new_')):metrics[f'cell{cell}.{selection}.{field}']=value
    for panel in ('old','new'):
        families[panel]={}
        for comparison in COMPARISONS:
            d={c:metrics[f'{panel}.{comparison}.cell{c}'] for c in CELLS}
            w=d['10']-d['00'];l=d['01']-d['00'];i=d['11']-d['10']-d['01']+d['00'];t=d['11']-d['00']
            comp={'weight_at_original':w,'location_at_deployed':l,'interaction':i,'total_sequential_minus_joint':t,
                  'weight_at_updated':d['11']-d['01'],'location_at_unit':d['11']-d['10'],'reconciliation_residual':t-(w+l+i)}
            require(abs(comp['reconciliation_residual'])<=1e-10,'decomposition reconciliation')
            require(comp==row['decompositions'][panel][comparison],'decomposition mismatch')
            families[panel][comparison]={'cells':d,'components':comp}
            for key,value in comp.items():metrics[f'{panel}.{comparison}.{key}']=value
    for name,values in row['baselines'].items():
        for panel,value in values.items():metrics[f'{panel}.baseline.{name}']=value
    require(len(metrics)==350 and metrics==row['metrics'],'saved metric family mismatch')
    require(row['fresh_context_checks']['fresh_unit_bitwise_equal'],'saved fresh equivalence false')
    require(row['fresh_context_checks']['direct_fresh_sha256']==row['fresh_step_sha256']['unit'],'saved fresh hash mismatch')
    for name,check in row['numerical_checks'].items():
        require(check['sequential_bitwise_equal'] and check['normalized_joint_math']['within_tolerance'],'saved parameter equivalence false')
        require(check['direct_sequential_sha256']==row['cells']['11']['branches'][name]['end_state_sha256'],'saved sequential endpoint mismatch')
        for panel,loss in check['loss_equivalence'].items():
            require(loss['within_tolerance'] and loss['difference']==loss['mathematical00_loss']-loss['actual00_loss'],'saved loss-equivalence arithmetic')
            require(loss['tolerance']==2e-5+2e-6*abs(loss['actual00_loss']) and abs(loss['difference'])<=loss['tolerance'],'saved loss-equivalence tolerance')
    return {'seed':row['seed'],'block':row['block'],'metrics':metrics,'families':families}

def summarize(records,key):
    values={s:[r['metrics'][key] for r in records if r['seed']==s] for s in SEEDS}
    require(all(len(v)==4 for v in values.values()),'four dependent states per seed required')
    means=np.array([np.mean(values[s]) for s in SEEDS],np.float64)
    rng=np.random.Generator(np.random.PCG64(7310301))
    boot=means[rng.integers(4,size=(10000,4))].mean(axis=1)
    signs=lambda vals:{'negative':sum(float(v)<0 for v in vals),'zero':sum(float(v)==0 for v in vals),'positive':sum(float(v)>0 for v in vals)}
    return {'mean':float(means.mean()),'ci95':[float(v) for v in np.percentile(boot,[2.5,97.5])],
            'seed_means':{str(s):float(v) for s,v in zip(SEEDS,means)},'state_sign_counts':signs([r['metrics'][key] for r in records]),
            'seed_sign_counts':signs(means),'cluster_count':4,'state_count':16,'status':'posthoc_descriptive_only',
            'warning':'few historical seed clusters; not fresh confirmation, dataset uncertainty, or retention evidence'}

def check_csv(root,records,summaries):
    def read(name):
        with (root/'analysis'/name).open(newline='') as f:return list(csv.DictReader(f))
    for name,expected_count in (('all-350-summary-metrics.csv',350),('aggregate-components.csv',88)):
        rows=read(name);require(len(rows)==expected_count,'CSV family count')
        seen=set()
        for row in rows:
            key=row.get('metric') or '.'.join(row[k] for k in ('panel','comparison','contrast'))
            require(key not in seen,'duplicate CSV metric');seen.add(key);s=summaries[key]
            for col,value in [('mean',s['mean']),('ci95_low',s['ci95'][0]),('ci95_high',s['ci95'][1])]+[(f'seed_{seed}',s['seed_means'][str(seed)]) for seed in SEEDS]:
                require(float(row[col])==value,'CSV scalar mismatch')
            for level in ('state','seed'):
                for sign in ('negative','positive','zero'):
                    require(int(row[('states' if level=='state' else 'seeds')+'_'+sign])==s[level+'_sign_counts'][sign],'CSV sign mismatch')
    lookup={(r['seed'],r['block']):r for r in records}
    for name,count in (('signed-state-components.csv',128),('signed-seed-components.csv',32)):
        rows=read(name);require(len(rows)==count,'CSV signed family count');seen=set()
        for row in rows:
            seed=int(row['seed']);panel=row['panel'];comparison=row['comparison'];block=int(row['block']) if 'block' in row else None
            ident=(seed,block,panel,comparison);require(ident not in seen,'duplicate signed CSV row');seen.add(ident)
            for key,value in row.items():
                if key in ('seed','block','panel','comparison'):continue
                component='cell'+key[1:] if key.startswith('D') else key
                expected=lookup[seed,block]['metrics'][f'{panel}.{comparison}.{component}'] if block is not None else summaries[f'{panel}.{comparison}.{component}']['seed_means'][str(seed)]
                require(float(value)==expected,'signed CSV component mismatch')

def verify(root):
    root=Path(root).resolve();manifest=inventory(root)
    receipt=load_json(root/'evidence/COMPLETE.json')
    require(sha(root/'evidence/COMPLETE.json')==COMPLETE,'original completion marker identity')
    expected={'run-input-freeze.json','summary.json'}|{f'seed-{s}-block-{b}.json' for s,b in itertools.product(SEEDS,BLOCKS)}
    require(set(receipt['files'])==expected,'original completion inventory mismatch')
    total=(root/'evidence/COMPLETE.json').stat().st_size
    for name,entry in receipt['files'].items():
        total+=entry['bytes']
        if name=='run-input-freeze.json':
            require(entry=={'bytes':18474,'sha256':FREEZE},'original undistributed freeze commitment changed');continue
        p=root/'evidence'/name
        require(p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'original retained evidence hash mismatch')
    require(total==receipt['artifact_bytes_including_complete']==3175082,'original nineteen-artifact accounting mismatch')
    freeze=load_json(root/'evidence/run-input-freeze.public.json')
    require(freeze['publication_derivative']['original_sha256']==FREEZE,'public freeze derivative binding')
    require(len(freeze['files'])==65 and set(freeze['files'])=={f'input-{i:03d}' for i in range(65)},'public freeze alias inventory')
    require(sha(root/'evidence/summary.json')==SUMMARY,'original summary identity')
    require((root/'evidence/execution-command.exit.txt').read_text().strip()=='0','saved exit receipt not zero')
    require((root/'evidence/execution-command.stderr.txt').stat().st_size==0,'saved scientific stderr not empty')
    records=[];raw=[]
    for seed,block in itertools.product(SEEDS,BLOCKS):
        row=load_json(root/f'evidence/seed-{seed}-block-{block}.json');raw.append(row)
        state=next(s for s in freeze['state_identifiers'] if (s['seed'],s['block'])==(seed,block))
        require(row['seed']==seed and row['block']==block and row['checkpoint_sha256']==state['checkpoint_sha256'],'state checkpoint identity mismatch')
        require(row['zero_based_update_offset']==state['zero_based_update_offset'],'state update offset mismatch')
        records.append(recompute_row(row))
    saved=load_json(root/'evidence/summary.json')
    summaries={key:summarize(records,key) for key in sorted(records[0]['metrics'])}
    compare_summary(saved['summaries'],summaries)
    require(saved['signed_state_metrics']==[{k:r[k] for k in ('seed','block','metrics')} for r in records],'saved signed metrics mismatch')
    require(saved['state_identities']==[[s,b] for s,b in itertools.product(SEEDS,BLOCKS)],'summary state inventory mismatch')
    require(saved['compute_graph_totals']=={k:sum(r['compute_graph'][k] for r in raw) for k in ('gradient_calls','candidate_example_forwards','panel_example_forwards')},'compute ledger mismatch')
    require(saved['compute_graph_totals']=={'gradient_calls':672,'candidate_example_forwards':1600,'panel_example_forwards':2408000},'compute totals changed')
    analysis=load_json(root/'analysis/saved-analysis.public.json')
    compare_summary(analysis['summaries'],summaries)
    require(analysis['records']==records,'full saved analysis records mismatch')
    check_csv(root,records,summaries)
    transforms=load_json(root/'transformations.json')
    for entry in transforms['entries']:
        p=root/entry['public_path'];require({'bytes':p.stat().st_size,'sha256':sha(p)}==entry['public'],'transformation public hash mismatch')
        if entry['kind']=='exact-byte-copy':require(entry['original']==entry['public'],'exact copy relabeled derivative')
    d00=summaries['old.mir_minus_highloss.cell00'];adverse=summaries['cell00.mir.old_change_from_original']
    require(adverse['state_sign_counts']['positive']==16 and d00['ci95'][1]>-.005,'adverse/failed-margin result changed')
    return {'status':'PASS_saved_evidence_only','recomputed_summaries':len(summaries),'states':16,'clusters':4,
            'public_files_including_allowlist':len(manifest['files'])+1,'original_run_artifacts':19,'original_run_bytes_including_complete':total,
            'original_freeze_sha256_commitment':FREEZE,'public_freeze_is_derivative':True,'scientific_execution_performed':False,
            'losses_gradients_or_parameter_arrays_recomputed':False,'execution_independently_attested':False,
            'model_code_imported':False,'numerical_software':'NumPy '+np.__version__,
            'old_joint_mir_change_from_original':adverse['mean'],'old_joint_mir_minus_highloss':d00,
            'old_interaction':summaries['old.mir_minus_highloss.interaction'],
            'limitations':'Saved scalar arithmetic and integrity only. Missing historical execution inputs and private provenance prevent full scientific rerun verification. Four reused historical clusters; not independent confirmation, retention efficacy or biological mapping.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    try:print(json.dumps(verify(Path(__file__).resolve().parent),sort_keys=True,indent=2,allow_nan=False))
    except (VerificationError,KeyError,StopIteration) as exc:raise SystemExit('FAIL: '+str(exc))

if __name__=='__main__':main()
