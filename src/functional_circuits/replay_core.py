"""Pure NumPy primitives for replay_digits_v1, without dataset or test access."""
import hashlib
import numpy as np

ARMS = ('reservoir', 'fifo', 'current_only')


def initialize(rng):
    return [rng.uniform(-np.sqrt(6/96), np.sqrt(6/96), (64, 32)), np.zeros(32),
            rng.uniform(-np.sqrt(6/42), np.sqrt(6/42), (32, 10)), np.zeros(10)]


def logits(parameters, x):
    w1, b1, w2, b2 = parameters
    return np.tanh(x @ w1 + b1) @ w2 + b2


def predict(parameters, x):
    return logits(parameters, x).argmax(axis=1)


def loss_gradient(parameters, x, y):
    w1, b1, w2, b2 = parameters
    h = np.tanh(x @ w1 + b1)
    z = h @ w2 + b2
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    normalizer = exp.sum(axis=1, keepdims=True)
    loss = np.mean(np.log(normalizer[:, 0]) - z[np.arange(len(y)), y])
    dz = exp / normalizer
    dz[np.arange(len(y)), y] -= 1
    dz /= len(y)
    dh = (dz @ w2.T) * (1-h*h)
    return float(loss), [x.T @ dh, dh.sum(axis=0), h.T @ dz, dz.sum(axis=0)]


class Memory:
    def __init__(self, arm, capacity, rng):
        if arm not in ARMS or type(capacity) is not int or capacity < 1:
            raise ValueError('Invalid memory configuration')
        self.arm, self.capacity, self.rng = arm, capacity, rng
        self.entries, self.positions, self.seen = [], [], 0

    def add(self, index):
        self.seen += 1
        if self.arm == 'current_only':
            return
        if len(self.entries) < self.capacity:
            self.entries.append(index)
            self.positions.append(self.seen)
            return
        slot = ((self.seen-1) % self.capacity if self.arm == 'fifo'
                else int(self.rng.integers(self.seen)))
        if slot < self.capacity:
            self.entries[slot], self.positions[slot] = index, self.seen


def train_stream(x, y, row_ids, seed, rate, arm):
    """Accept training arrays ONLY. Return all five snapshots, never score holdouts.

    Buffer entries are indices into these arrays. Global row IDs are diagnostic;
    no holdout array or task identifier is ever supplied to the model.
    """
    x, y, row_ids = np.asarray(x, dtype=np.float64), np.asarray(y), np.asarray(row_ids)
    if (x.ndim != 2 or x.shape != (len(y), 64) or y.ndim != 1 or row_ids.shape != y.shape
            or y.dtype.kind not in 'iu' or row_ids.dtype.kind not in 'iu'
            or len(set(row_ids.tolist())) != len(y) or not np.isfinite(x).all()
            or np.any(x < 0) or np.any(x > 1) or set(y.tolist()) != set(range(10))
            or type(seed) is not int or seed < 0 or not np.isfinite(rate) or rate <= 0
            or arm not in ARMS):
        raise ValueError('Invalid training data or configuration')
    rngs = [np.random.default_rng(np.random.SeedSequence([seed, i])) for i in range(5)]
    order = rngs[0].permutation(10)
    parameters = initialize(rngs[2])
    initial_hash = hashlib.sha256(b''.join(w.astype('<f8').tobytes() for w in parameters)).hexdigest()
    memory = Memory(arm, 100, rngs[3])
    counts = dict(fresh_observations=0, fresh_presentations=0, replay=0, presentations=0, updates=0)
    replay_counts, fresh_counts = np.zeros(len(y), dtype=int), np.zeros(len(y), dtype=int)
    snapshots, boundaries = [], []
    fallback = 0
    for block in range(5):
        labels = order[2*block:2*block+2]
        stream = rngs[1].permutation(np.flatnonzero(np.isin(y, labels)))
        for start in range(0, len(stream), 16):
            fresh = stream[start:start+16]
            m = len(fresh)
            for _ in range(4):
                pool = np.asarray(memory.entries, dtype=int) if memory.entries else fresh
                if not memory.entries:
                    fallback += 1
                replay = pool[rngs[4].integers(len(pool), size=m)]
                np.add.at(replay_counts, replay, 1)
                indices = np.concatenate([fresh, replay])
                loss, gradients = loss_gradient(parameters, x[indices], y[indices])
                if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients):
                    raise FloatingPointError('Nonfinite loss or gradient; no study retuning allowed')
                for p, g in zip(parameters, gradients):
                    p -= rate * g
                counts['fresh_presentations'] += m
                counts['replay'] += m
                counts['presentations'] += 2*m
                counts['updates'] += 1
            # Admission occurs strictly AFTER all four updates for this batch.
            for index in fresh:
                memory.add(int(index))
            fresh_counts[fresh] += 1
            counts['fresh_observations'] += m
        snapshots.append([p.copy() for p in parameters])
        ages = [memory.seen - position for position in memory.positions]
        boundaries.append(dict(block=block, labels=labels.tolist(), counts=counts.copy(),
            memory_row_ids=row_ids[memory.entries].tolist(), memory_entry_ages=ages,
            memory_class_counts=np.bincount(y[memory.entries], minlength=10).tolist(),
            replay_class_counts=np.bincount(y, weights=replay_counts, minlength=10).astype(int).tolist(),
            empty_buffer_fallback_updates=fallback))
    if not np.all(fresh_counts == 1) or counts['replay'] != 4*len(y):
        raise AssertionError('Exposure-count invariant failed')
    return dict(arm=arm, seed=seed, rate=float(rate), label_order=order.tolist(),
                initial_sha256=initial_hash, counts=counts, snapshots=snapshots,
                boundaries=boundaries, fresh_counts=fresh_counts.tolist(),
                replay_counts=replay_counts.tolist(), replayed_row_ids=row_ids[replay_counts > 0].tolist(),
                parameter_bytes=sum(p.nbytes for p in parameters),
                gradient_bytes=sum(p.nbytes for p in parameters),
                optimizer_state_bytes=0,
                memory_logical_payload_bytes=len(memory.entries)*(64*8+8),
                memory_note='Logical float64 features/int64 labels; index-backed implementation stores IDs and references the resident training arrays. Excludes Python/container overhead; measure peak RSS separately.')


def score(predictions, labels):
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    if (predictions.shape != labels.shape or labels.ndim != 1
            or set(labels.tolist()) != set(range(10))
            or predictions.dtype.kind not in 'iu' or labels.dtype.kind not in 'iu'
            or np.any(predictions < 0) or np.any(predictions > 9)):
        raise ValueError('Scores require ten-class labeled data and valid predictions')
    recall = [float(np.mean(predictions[labels == c] == c)) for c in range(10)]
    return dict(macro_accuracy=float(np.mean(recall)), micro_accuracy=float(np.mean(predictions == labels)),
                per_class_recall=recall)


def forgetting(block_accuracy):
    """Rows: five boundaries; columns: five blocks in that run's label order."""
    a = np.asarray(block_accuracy, dtype=float)
    if a.shape != (5, 5) or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
        raise ValueError('Expected a finite five-by-five accuracy matrix')
    return float(np.mean([np.max(a[i:4, i])-a[4, i] for i in range(4)]))


def choose_rate(candidate_scores):
    """One common rate selected on development validation scores only."""
    if set(candidate_scores) != {.03, .1, .3}:
        raise ValueError('Protocol requires exactly three specified rates')
    values = {r: np.asarray(v, dtype=float) for r,v in candidate_scores.items()}
    if any(
            v.shape != (6,) or not np.isfinite(v).all() or np.any(v < 0) or np.any(v > 1)
            for v in values.values()):
        raise ValueError('Invalid candidate scores')
    means = {r: float(v.mean()) for r,v in values.items()}
    best = max(means.values())
    return min(r for r, value in means.items() if value >= best-1e-12)


def paired_summary(differences):
    d = np.asarray(differences, dtype=float)
    if d.shape != (10,) or not np.isfinite(d).all() or np.any(np.abs(d) > 1):
        raise ValueError('Expected ten finite paired accuracy differences')
    rng = np.random.default_rng(9801)
    means = d[rng.integers(10, size=(10000, 10))].mean(axis=1)
    interval = np.percentile(means, [2.5, 97.5]).tolist()
    return dict(mean_difference=float(d.mean()), interval_95=interval,
                supported_for_setup=bool(d.mean() >= .02 and interval[0] > 0),
                interval_scope='Descriptive seed bootstrap, conditional on one dataset and fixed setup')
