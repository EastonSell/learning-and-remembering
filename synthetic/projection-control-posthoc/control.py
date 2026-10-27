"""Deterministic Euclidean projection control; not a replay experiment."""
from math import fsum, hypot, isclose, isfinite


def project(h, gradients, rank_rtol=1e-12):
    """Return h projected off the numerical span of row gradients.

    Twice-reorthogonalized modified Gram-Schmidt; vectors whose residual norm
    is at most rank_rtol times their original norm are treated as dependent.
    """
    h = tuple(float(x) for x in h)
    if not h or not all(isfinite(x) for x in h):
        raise ValueError('h must be a nonempty finite vector')
    if not isfinite(rank_rtol) or not 0 <= rank_rtol < 1:
        raise ValueError('rank_rtol must be finite and in [0, 1)')
    basis = []
    for row in gradients:
        row = tuple(float(x) for x in row)
        if len(row) != len(h) or not all(isfinite(x) for x in row):
            raise ValueError('gradient dimensions must match h and values must be finite')
        norm = hypot(*row)
        if norm == 0:
            continue
        v = list(row)
        for _ in range(2):
            for q in basis:
                dot = fsum(a*b for a, b in zip(v, q))
                v = [a - dot*b for a, b in zip(v, q)]
        residual = hypot(*v)
        if residual > rank_rtol * norm and len(basis) < len(h):
            basis.append(tuple(x / residual for x in v))
    d = list(h)
    for _ in range(2):
        for q in basis:
            dot = fsum(a*b for a, b in zip(d, q))
            d = [a - dot*b for a, b in zip(d, q)]
    return tuple(d)


def old(x, y):
    return x + y*y/2


def report(checks):
    passed = 0
    for name, actual, expected in checks:
        ok = isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        passed += ok
        print(f'{"PASS" if ok else "FAIL"} {name}: {actual:.17g} (expected {expected:.17g})')
    print(f'{passed}/{len(checks)} deterministic assertions passed')
    return 0 if passed == len(checks) else 1


def main():
    eta = 0.1
    h, g = (1.0, 1.0), (1.0, 0.0)
    d = project(h, [g])
    full = project(h, [(1., 0.), (0., 1.)])
    checks = [
        ('stored_gradient_orthogonality', fsum(a*b for a,b in zip(g,d)), 0.0),
        ('unprojected_old_loss_change', old(*(-eta*x for x in h)) - old(0,0), -0.095),
        ('projected_old_loss_change', old(*(-eta*x for x in d)) - old(0,0), 0.005),
        ('full_rank_update_norm', hypot(*full), 0.0),
    ]
    return report(checks)

if __name__ == '__main__':
    raise SystemExit(main())
