"""Illustrate forgetting, rehearsal and two replay scores; not benchmark evidence.

Python standard library only. No files, random sampling, downloads or training
libraries. Both toy tasks share an exact solution: weights (1, -1).
"""
import json

TASK_A = ((1.0, 0.0), 1.0)
TASK_B = ((1.0, 1.0), 0.0)
RATE = 0.25


def loss(weights, example):
    """Half-squared error for a two-feature linear prediction."""
    x, target = example
    error = sum(w * value for w, value in zip(weights, x)) - target
    return 0.5 * error * error


def step(weights, examples):
    """One SGD step on the MEAN loss; return new weights without mutation."""
    gradient = [0.0, 0.0]
    for x, target in examples:
        error = sum(w * value for w, value in zip(weights, x)) - target
        for j in range(2):
            gradient[j] += error * x[j] / len(examples)
    return tuple(w - RATE * g for w, g in zip(weights, gradient))


def run():
    """Compare equal-size models, updates and presentations, not equal history."""
    weights = (0.0, 0.0)
    for _ in range(40):
        weights = step(weights, (TASK_A, TASK_A))
    result = {'after_a': {'weights': list(weights),
                          'a_loss': loss(weights, TASK_A),
                          'b_loss': loss(weights, TASK_B)}}
    for name, batch in [('current_only', (TASK_B, TASK_B)),
                        ('replay', (TASK_B, TASK_A))]:
        current = weights
        for _ in range(120):
            current = step(current, batch)
        result[name] = {'weights': list(current),
                        'a_loss': loss(current, TASK_A),
                        'b_loss': loss(current, TASK_B),
                        'updates': 120, 'presentations': 240,
                        'stored_old_examples': int(name == 'replay')}
    return result


def selection_example():
    """Show hard-now versus threatened-next rankings on three eligible items.

All labels obey y=x1-x2. These are scoring illustrations, NOT an implementation
of the Fashion-MNIST quotas, model, cross-entropy or complete MIR algorithm.
The virtual incoming-data step is discarded rather than committed.
"""
    weights = (1.0, 0.0)
    virtual = step(weights, (TASK_B,))
    candidates = [('old_anchor', TASK_A),
                  ('hard_now', ((0.0, 1.0), -1.0)),
                  ('easy_but_threatened', ((2.0, 0.0), 2.0))]
    rows = []
    for name, example in candidates:
        now = loss(weights, example)
        after = loss(virtual, example)
        rows.append({'name': name, 'loss_now': now,
                     'loss_after_virtual_step': after,
                     'loss_increase': after - now})
    return {'weights_before': list(weights), 'virtual_weights': list(virtual),
            'candidates': rows,
            'high_loss_choice': max(rows, key=lambda r: r['loss_now'])['name'],
            'predicted_harm_choice': max(rows, key=lambda r: r['loss_increase'])['name'],
            'uniform_probabilities': [1/3] * 3}


def main():
    print('ILLUSTRATION only: not benchmark evidence or a biological simulation.')
    print(json.dumps({'forgetting_and_rehearsal': run(),
                      'selection_scores': selection_example()}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
