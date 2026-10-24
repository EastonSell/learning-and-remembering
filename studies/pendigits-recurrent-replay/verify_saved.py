"""Check published text evidence and recalculate results; no training/downloads."""
import hashlib
import json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent
catalog = json.loads((P / 'compact-evidence-manifest.json').read_text())
for name, expected in catalog.items():
    assert hashlib.sha256((P / name).read_bytes()).hexdigest() == expected, name
freeze = json.loads((P / 'freeze.json').read_text())
review = json.loads((P / 'independent-review.json').read_text())
for key, name in [('source_sha256', 'experiment.py'), ('protocol_sha256', 'PROTOCOL.md')]:
    assert catalog[name] == freeze[key] == review[key]
assert catalog['independent-review.json'] == freeze['review_sha256']
rows = json.loads((P / 'results/metrics.json').read_text())
summary = json.loads((P / 'results/summary.json').read_text())
arms = ['uniform', 'high_loss', 'mir']
seeds = list(range(31001, 31011))
by = {(r['seed'], r['arm']): r for r in rows}
assert len(rows) == len(by) == 30
assert set(by) == {(s, a) for s in seeds for a in arms}
for s in seeds:
    counts = []
    for a in arms:
        row = by[s, a]
        assert len(row['boundaries']) == 10
        for i, boundary in enumerate(row['boundaries']):
            assert boundary['phase'] == i
            assert len(boundary['recalls']) == 10
            recalls = np.array(boundary['recalls'])
            assert np.isfinite(recalls).all() and ((0 <= recalls) & (recalls <= 1)).all()
            assert float(recalls.mean()) == boundary['macro']
            assert float(recalls[boundary['current_pair']].mean()) == boundary['acquisition']
        assert row['final_macro'] == row['boundaries'][-1]['macro']
        assert row['mean_acquisition'] == np.mean([b['acquisition'] for b in row['boundaries']])
        counts.append(row['counts_per_arm'])
    assert counts[0] == counts[1] == counts[2]
    assert counts[0] == dict(actual_steps=760, virtual_steps=760,
        candidate_pre_post_scored=192000, fresh_presentations=23980,
        replay_presentations=24192)
d = np.array([by[s, 'mir']['final_macro'] - by[s, 'uniform']['final_macro'] for s in seeds])
acq = np.array([by[s, 'mir']['mean_acquisition'] - by[s, 'uniform']['mean_acquisition'] for s in seeds])
ci = np.quantile(np.random.default_rng(20261025).choice(d, size=(10000, 10), replace=True).mean(1), [.025, .975])
assert np.array_equal(d, summary['mir_minus_uniform'])
assert np.array_equal(ci, summary['paired_seed_bootstrap_95'])
assert d.mean() == summary['mean_difference']
assert acq.mean() == summary['mean_acquisition_difference']
assert {a: float(np.mean([by[s, a]['final_macro'] for s in seeds])) for a in arms} == summary['mean_final_macro']
assert bool(d.mean() >= .01 and ci[0] > 0 and acq.mean() >= -.02) == summary['support']
assert summary['support'] is False
print(json.dumps({'status': 'PASS', 'cases': 30, 'boundaries': 300,
    'scope': 'published text hashes, saved recall arithmetic, paired bootstrap and accounting only; no inference or training',
    'mir_minus_uniform_pp': float(100*d.mean()), 'ci95_pp': (100*ci).tolist(),
    'acquisition_difference_pp': float(100*acq.mean()), 'support': summary['support']}, indent=2))
