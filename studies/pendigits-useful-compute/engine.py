"""Deterministic useful-compute mechanics; no data I/O or automatic execution."""
import hashlib
import itertools
import math
import time
import numpy as np

ARMS = ('uniform_base', 'mir', 'uniform_credit')
FORWARD_UNITS = 13
TRAIN_UNITS = 31
EXTRA_COST = TRAIN_UNITS * 32
LR = .05


def init(seed):
    r = np.random.default_rng(seed)
    return [r.normal(size=(16, 64))*np.sqrt(2/16), np.zeros(64),
            r.normal(size=(64, 10))*np.sqrt(2/64), np.zeros(10)]


def forward(w, x):
    h = np.maximum(0, x @ w[0] + w[1])
    z = h @ w[2] + w[3]
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    if not np.isfinite(p).all():
        raise ValueError('nonfinite probabilities')
    return h, p


def losses(w, x, y):
    return -np.log(np.maximum(forward(w, x)[1][np.arange(len(y)), y], 1e-300))


def step(w, x, y):
    h, p = forward(w, x)
    p[np.arange(len(y)), y] -= 1
    p /= len(y)
    dh = (p @ w[2].T) * (h > 0)
    grads = [x.T @ dh, dh.sum(0), h.T @ p, p.sum(0)]
    if not all(np.isfinite(g).all() for g in grads):
        raise ValueError('nonfinite gradient')
    result = [v - LR*g for v, g in zip(w, grads)]
    if not all(np.isfinite(v).all() for v in result):
        raise ValueError('nonfinite weights')
    return result


def metrics(w, x, y):
    p = forward(w, x)[1]
    pred = p.argmax(1)
    recalls = [float(np.mean(pred[y == c] == c)) for c in range(10)]
    if not all(math.isfinite(v) for v in recalls):
        raise ValueError('every test class must exist')
    return {'macro': float(np.mean(recalls)), 'micro': float(np.mean(pred == y)),
            'recalls': recalls,
            'ce': float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-300)).mean())}, pred, p


def build_phases(y, train, seed):
    rng = np.random.default_rng(seed + 100000)
    parts = {}
    for c in range(10):
        ids = train[y[train] == c].copy()
        rng.shuffle(ids)
        parts[c] = np.array_split(ids, 2)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    phases = []
    for visit, ps in enumerate([pairs, list(reversed(pairs))]):
        for pair in ps:
            ids = np.concatenate([parts[c][visit] for c in pair])
            rng.shuffle(ids)
            phases.append((pair, ids))
    return phases


def case_order(seed_index):
    return tuple(itertools.permutations(ARMS))[seed_index % 6]


def train_case(x, y, phases, seed, arm, callback=None, check=lambda: None):
    """Run one arm; callback receives snapshots outside the training timer.

    This primitive supports synthetic tests. Real-data callers must satisfy the
    separate run.py admission/data/protocol contract before passing arrays here.
    """
    if arm not in ARMS:
        raise ValueError('unknown arm')
    started = time.perf_counter()
    segment = started
    train_seconds = 0.
    w = init(seed)
    buf, seen, credit = [], 0, 0
    reservoir_rng = np.random.default_rng(seed + 200000)
    candidate_rng = np.random.default_rng(seed + 300000)
    base_rng = np.random.default_rng(seed + 400000)
    extra_rng = np.random.default_rng(seed + 500000)
    base_trace, reservoir_trace = hashlib.sha256(), hashlib.sha256()
    count = dict(base_steps=0, extra_steps=0, virtual_steps=0,
                 fresh_presentations=0, base_replay_presentations=0,
                 extra_replay_presentations=0, virtual_fresh_presentations=0,
                 candidate_scoring_rows=0, work_units=0, credited_units=0)

    def train(weights, ids, kind):
        check()
        result = step(weights, x[ids], y[ids])
        count['work_units'] += TRAIN_UNITS * len(ids)
        count[kind + '_steps'] += 1
        check()
        return result

    def score(weights, ids):
        check()
        result = losses(weights, x[ids], y[ids])
        count['candidate_scoring_rows'] += len(ids)
        count['work_units'] += FORWARD_UNITS * len(ids)
        check()
        return result

    def pick(rng):
        return np.asarray(buf, dtype=int)[rng.choice(len(buf), size=32, replace=len(buf) < 32)]

    for phase, (pair, ids) in enumerate(phases):
        for start in range(0, len(ids), 32):
            fresh = ids[start:start+32]
            for _ in range(4):
                check()
                if buf:
                    if arm == 'mir':
                        candidates = np.asarray(buf, dtype=int)[candidate_rng.choice(
                            len(buf), size=min(128, len(buf)), replace=False)]
                        virtual = train(w, fresh, 'virtual')
                        count['virtual_fresh_presentations'] += len(fresh)
                        pre, post = score(w, candidates), score(virtual, candidates)
                        rank = np.argsort(-(post-pre), kind='stable')
                        selected = candidates[np.resize(rank[:32], 32)]
                    else:
                        selected = pick(base_rng)
                    joint = np.concatenate([fresh, selected])
                    count['base_replay_presentations'] += 32
                    base_trace.update(np.asarray(selected, dtype='<i8').tobytes())
                else:
                    joint = fresh
                w = train(w, joint, 'base')
                count['fresh_presentations'] += len(fresh)
                if arm == 'uniform_credit' and buf:
                    allowance = TRAIN_UNITS*len(fresh) + 2*FORWARD_UNITS*min(128, len(buf))
                    credit += allowance
                    count['credited_units'] += allowance
                    while credit >= EXTRA_COST:
                        selected = pick(extra_rng)
                        w = train(w, selected, 'extra')
                        count['extra_replay_presentations'] += 32
                        credit -= EXTRA_COST
                if credit < 0 or credit >= EXTRA_COST:
                    raise ValueError('credit ledger invariant')
            for idx in fresh:
                seen += 1
                if len(buf) < 256:
                    buf.append(int(idx))
                else:
                    pos = reservoir_rng.integers(seen)
                    if pos < 256:
                        buf[pos] = int(idx)
            reservoir_trace.update(np.asarray(buf, dtype='<i8').tobytes())
        check()
        train_seconds += time.perf_counter() - segment
        partial = {'counts': count.copy(), 'credit': credit, 'train_seconds': train_seconds}
        if callback is not None:
            callback(phase, pair, [v.copy() for v in w], partial)
        check()
        segment = time.perf_counter()
    train_seconds += time.perf_counter() - segment
    count['base_replay_trace_sha256'] = base_trace.hexdigest()
    count['reservoir_trace_sha256'] = reservoir_trace.hexdigest()
    return {'weights': w, 'counts': count, 'credit': credit,
            'train_seconds': train_seconds, 'case_seconds': time.perf_counter()-started}


def validate_accounting(rows, seeds):
    by = {(r['seed'], r['arm']): r for r in rows}
    for seed in seeds:
        b, m, u = [by[seed, a] for a in ARMS]
        for r in (b, m, u):
            c = r['counts']
            integer_fields = set(c) - {'base_replay_trace_sha256', 'reservoir_trace_sha256'}
            if any(type(c[k]) is not int or c[k] < 0 for k in integer_fields):
                raise ValueError('invalid accounting integer')
            actual = c['fresh_presentations'] + c['base_replay_presentations'] + c['extra_replay_presentations']
            expected = TRAIN_UNITS*(actual+c['virtual_fresh_presentations']) + FORWARD_UNITS*c['candidate_scoring_rows']
            if c['work_units'] != expected or c['extra_replay_presentations'] != 32*c['extra_steps']:
                raise ValueError('executed work mismatch')
            if c['base_steps'] != b['counts']['base_steps'] or c['fresh_presentations'] != b['counts']['fresh_presentations'] or c['base_replay_presentations'] != b['counts']['base_replay_presentations']:
                raise ValueError('base exposure mismatch')
            if c['reservoir_trace_sha256'] != b['counts']['reservoir_trace_sha256']:
                raise ValueError('reservoir mismatch')
        for r in (b, u):
            if any(r['counts'][k] for k in ('virtual_steps', 'virtual_fresh_presentations', 'candidate_scoring_rows')):
                raise ValueError('uniform performed unused selector work')
        if b['counts']['extra_steps'] or m['counts']['extra_steps']:
            raise ValueError('unplanned extra steps')
        if b['counts']['base_replay_trace_sha256'] != u['counts']['base_replay_trace_sha256']:
            raise ValueError('base uniform trace mismatch')
        if not 0 <= u['credit'] < EXTRA_COST:
            raise ValueError('unspent credit outside range')
        mir_overhead = TRAIN_UNITS*m['counts']['virtual_fresh_presentations'] + FORWARD_UNITS*m['counts']['candidate_scoring_rows']
        if mir_overhead != u['counts']['credited_units']:
            raise ValueError('credit allowance mismatch')
        if m['counts']['work_units']-u['counts']['work_units'] != u['credit']:
            raise ValueError('credit reconciliation mismatch')


def analyze(rows, seeds, bootstrap_seed=20261027, replicates=10000):
    expected = {(s, a) for s in seeds for a in ARMS}
    by = {(r['seed'], r['arm']): r for r in rows}
    if len(seeds) != 12 or len(set(seeds)) != 12 or len(rows) != len(expected) or set(by) != expected:
        raise ValueError('complete fixed paired denominator required')
    for r in rows:
        if r['status'] != 'complete' or len(r['boundaries']) != 10:
            raise ValueError('incomplete case or boundary denominator')
        for key in ('final_macro', 'mean_acquisition', 'train_seconds'):
            if not math.isfinite(r[key]):
                raise ValueError('nonfinite outcome')
        if not 0 <= r['final_macro'] <= 1 or not 0 <= r['mean_acquisition'] <= 1 or r['train_seconds'] <= 0:
            raise ValueError('invalid accuracy or runtime')
        if any(not math.isfinite(b[k]) or not 0 <= b[k] <= 1 for b in r['boundaries'] for k in ('macro', 'acquisition')):
            raise ValueError('invalid boundary metric')
        if not math.isclose(r['final_macro'], r['boundaries'][-1]['macro'], abs_tol=1e-12) or not math.isclose(r['mean_acquisition'], np.mean([b['acquisition'] for b in r['boundaries']]), abs_tol=1e-12):
            raise ValueError('summary and boundary mismatch')
    ix = np.random.default_rng(bootstrap_seed).integers(0, len(seeds), size=(replicates, len(seeds)))
    result = {'planned_cases': len(expected), 'completed_cases': len(rows), 'comparisons': {}}
    for comparator in ('uniform_credit', 'uniform_base'):
        pairs = []
        for seed in seeds:
            m, u = by[seed, 'mir'], by[seed, comparator]
            pairs.append({'seed': seed, 'macro_difference': m['final_macro']-u['final_macro'],
                          'acquisition_difference': m['mean_acquisition']-u['mean_acquisition'],
                          'train_seconds_difference': m['train_seconds']-u['train_seconds'],
                          'mir_train_seconds': m['train_seconds'], 'comparator_train_seconds': u['train_seconds'],
                          'runtime_ratio': m['train_seconds']/u['train_seconds']})
        vals = {k: np.array([p[k] for p in pairs]) for k in
                ('macro_difference', 'acquisition_difference', 'train_seconds_difference')}
        logratio = np.log([p['runtime_ratio'] for p in pairs])
        summary = {'pairs': pairs}
        for k, v in vals.items():
            summary['mean_'+k] = float(v.mean())
            summary[k+'_95'] = np.quantile(v[ix].mean(1), [.025, .975], method='linear').tolist()
        summary['geomean_runtime_ratio'] = float(np.exp(logratio.mean()))
        summary['runtime_ratio_95'] = np.exp(np.quantile(logratio[ix].mean(1), [.025, .975], method='linear')).tolist()
        support = (summary['mean_macro_difference'] >= .01 and summary['macro_difference_95'][0] > 0
                   and summary['mean_acquisition_difference'] >= -.02)
        summary['accuracy_support'] = bool(support) if comparator == 'uniform_credit' else None
        summary['within_environment_dominance'] = bool(support and summary['runtime_ratio_95'][1] < 1) if comparator == 'uniform_credit' else None
        result['comparisons'][comparator] = summary
    return result
