"""Saved public scalar analysis and exact synthetic control; no model execution."""
import csv, hashlib, json, statistics
from fractions import Fraction as Q
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'inputs'

def toy(eta=Q(3,10), h=Q(4), linear_eval=False):
    a,b=Q(8,13),Q(5,13)
    # F(x)=(x+1)^2/2; R_M=2(x-.05)^2; R_H=2(x+.05)^2+1.
    # General h retains gradients at zero +/- .2; constants irrelevant to endpoints.
    def V(x): return x if linear_eval else (x+Q(1,20))**2/2
    out={}
    for k,alpha,beta,loc in [('00',a,b,0),('01',a,b,1),('10',Q(1),Q(1),0),('11',Q(1),Q(1),1)]:
        fresh=-eta*alpha
        endpoints={name:fresh-eta*beta*(r+h*fresh*loc) for name,r in [('mir',-Q(1,5)),('highloss',Q(1,5))]}
        out[k]={'endpoints':endpoints,'D':V(endpoints['mir'])-V(endpoints['highloss'])}
    out['interaction']=out['11']['D']-out['10']['D']-out['01']['D']+out['00']['D']
    return out

def summary(vals):
    return {'mean':statistics.fmean(vals),'negative':sum(x<0 for x in vals),'positive':sum(x>0 for x in vals)}

def main():
    manifest=json.loads((SOURCE/'source_manifest.json').read_text())
    known={x['path']:x for x in manifest['files']}
    rows=[]; sources=[]
    for p in sorted((SOURCE/'evidence').glob('seed-*-block-*.json')):
        data=p.read_bytes(); rel='studies/replay-update-decomposition/'+str(p.relative_to(SOURCE)); identity=known[rel]
        assert hashlib.sha256(data).hexdigest()==identity['sha256']
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==identity['git_blob']
        sources.append(identity)
        x=json.loads(data); d={k:v['branches']['mir']['old_loss']-v['branches']['highloss']['old_loss'] for k,v in x['cells'].items()}
        r={'seed':x['seed'],'block':x['block'],**{'D'+k:v for k,v in d.items()}}
        r['I']=d['11']-d['10']-d['01']+d['00']
        for selector in ('mir','highloss'):
            r[selector+'_unit_updated_change_original']=x['cells']['11']['branches'][selector]['old_loss']-x['baselines']['original']['old']
            for w in ('0','1'):
                r[selector+'_location_'+w]=x['cells'][w+'1']['branches'][selector]['old_loss']-x['cells'][w+'0']['branches'][selector]['old_loss']
        rows.append(r)
    assert {(r['seed'],r['block']) for r in rows}=={(s,b) for s in range(2005,2009) for b in range(4)} and len(rows)==16
    agg={k:summary([r[k] for r in rows]) for k in rows[0] if k not in ('seed','block')}
    seed={str(s):statistics.fmean(r['I'] for r in rows if r['seed']==s) for s in range(2005,2009)}
    largest=max(rows,key=lambda r:r['I'])
    result={'status':'posthoc_saved_arithmetic_not_new_model_evidence','aggregation':'four equal blocks per seed; four equal seeds','summary':agg,'seed_interaction_means':seed,'leave_one_seed_out_interaction':{str(s):statistics.fmean(r['I'] for r in rows if r['seed']!=s) for s in range(2005,2009)},'largest_state':largest,'largest_state_share_of_sum':largest['I']/sum(r['I'] for r in rows),'leave_largest_state_out_unbalanced_descriptive_only':statistics.fmean(r['I'] for r in rows if r is not largest),'both_selectors_benefit_unit_location':sum(r['mir_location_1']<0 and r['highloss_location_1']<0 for r in rows)}
    (HERE/'saved_results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (HERE/'state_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (HERE/'source_identities.json').write_text(json.dumps({'repo':'EastonSell/learning-and-remembering','source_commit':manifest['commit'],'unchanged_at_published_commit':'b0aeb63658c38b17e0e7ee42284dac724a75ee1c','basis':'Existing verified publication receipt records all frozen blobs unchanged; no new live-head claim.','files':sources},indent=2)+'\n')
    def convert(x):
        if isinstance(x,Q):return {'exact':str(x),'float':float(x)}
        if isinstance(x,dict):return {k:convert(v) for k,v in x.items()}
        return x
    cases={'quadratic_control':toy(),'affine_evaluator_control':toy(linear_eval=True),'constant_replay_gradient_control':toy(h=Q(0)),'half_eta':toy(Q(3,20)), 'evaluator_null_control':{'q_M0':[0,0], 'q_H0':[1,0], 'q_M1':[0,1], 'q_H1':[1,-1], 'evaluator':'V(x,y)=x*x/2', 'differential_drift_squared_norm':4, 'C00':Q(-1,2), 'C10':Q(-1,2), 'C01':Q(-1,2), 'C11':Q(-1,2), 'A':Q(0), 'B':Q(0)}}
    (HERE/'synthetic_results.json').write_text(json.dumps(convert(cases),indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
