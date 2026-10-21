"""Synthetic episodes. No downloaded or personal data."""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    sensory_y: np.ndarray
    memory_y: np.ndarray

    @property
    def context(self):
        return (self.x[:, -1] > 0).astype(int)

    def fingerprint(self):
        h = hashlib.sha256()
        for a in (self.x, self.y, self.sensory_y, self.memory_y):
            h.update(a.tobytes())
        return h.hexdigest()


def evidence_score(x, role):
    """Distinct nonlinear category rules, including one nuisance dimension."""
    if role == "sensory":
        return x[:, 0] + .6*x[:, 1] - .5*x[:, 2] + .35*x[:, 3]*x[:, 4]
    if role == "memory":
        return x[:, 0] - .6*x[:, 1] + .5*x[:, 2] + .35*x[:, 3]*x[:, 4]
    raise ValueError(role)


def evidence(rng, labels, role):
    x = rng.normal(size=(len(labels), 6))
    residual = evidence_score(x, role) - x[:, 0]
    x[:, 0] = (2*labels-1) * (np.abs(rng.normal(size=len(labels))) + .05) - residual
    return x


def retrieve(bank, query):
    """Exact episode-local lookup shared by ALL architectures, not learned memory."""
    return bank[np.arange(len(bank)), query].copy()


def generate(n, seed, agreement=.8):
    if n <= 0 or not 0 <= agreement <= 1:
        raise ValueError("Positive n and agreement in [0, 1] are required")
    rng = np.random.default_rng(seed)
    sy = rng.integers(0, 2, size=n)
    my = np.where(rng.random(n) < agreement, sy, 1-sy)
    sensory = evidence(rng, sy, "sensory")
    memory = evidence(rng, my, "memory")
    bank = rng.normal(size=(n, 4, 6))
    query = rng.integers(0, 4, size=n)
    bank[np.arange(n), query] = memory
    recalled = retrieve(bank, query)
    context = rng.integers(0, 2, size=n)
    x = np.column_stack([sensory, recalled, 2*context-1]).astype(np.float64)
    y = np.where(context == 1, my, sy).astype(np.float64)
    return Dataset(x, y, sy, my)


def memory_counterfactual(base, donor):
    if len(base.y) != len(donor.y):
        raise ValueError("Base and donor sizes must agree")
    return np.where(base.context == 1, donor.memory_y, base.sensory_y)
