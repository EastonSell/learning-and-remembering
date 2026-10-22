"""Numerical/protocol checks on synthetic fixtures, not a research run."""
import unittest
import numpy as np
from functional_circuits.replay_core import (initialize, loss_gradient, predict,
    Memory, train_stream, score, forgetting, choose_rate, paired_summary)


class ReplayCoreTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(1)
        y = np.repeat(np.arange(10), 9)
        return rng.random((90, 64)), y, np.arange(90) + 1000

    def test_every_parameter_gradient(self):
        rng = np.random.default_rng(10)
        p = initialize(rng)
        x, y = rng.random((3, 64)), np.array([1, 3, 7])
        _, gradients = loss_gradient(p, x, y)
        self.assertEqual(sum(w.size for w in p), 2410)
        eps = 1e-5
        for w, g in zip(p, gradients):
            for index in np.ndindex(w.shape):
                original = w[index]
                w[index] = original + eps
                plus = loss_gradient(p, x, y)[0]
                w[index] = original - eps
                minus = loss_gradient(p, x, y)[0]
                w[index] = original
                self.assertAlmostEqual(g[index], (plus-minus)/(2*eps), places=7)

    def test_softmax_stability_and_inference_is_nonmutating(self):
        p = initialize(np.random.default_rng(1))
        p[-1][:] = 10000
        before = [v.copy() for v in p]
        loss, gradient = loss_gradient(p, np.ones((2, 64)), np.array([0, 1]))
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.isfinite(g).all() for g in gradient))
        predict(p, np.ones((2, 64)))
        for a, b in zip(p, before):
            np.testing.assert_array_equal(a, b)

    def test_fifo_and_empty_buffer(self):
        m = Memory('fifo', 3, np.random.default_rng(0))
        self.assertEqual(m.entries, [])
        for i in range(8):
            m.add(i)
        self.assertEqual(set(m.entries), {5, 6, 7})
        self.assertEqual(sorted(m.positions), [6, 7, 8])

    def test_reservoir_inclusion_frequency_predeclared_bound(self):
        # 2,000 fixed trials, capacity 4 of 20. Expected count 400;
        # loose fixed bound +/-100 (>5 marginal standard deviations).
        counts = np.zeros(20, dtype=int)
        for seed in range(2000):
            m = Memory('reservoir', 4, np.random.default_rng(seed))
            for i in range(20):
                m.add(i)
            counts[m.entries] += 1
        self.assertTrue(np.all(np.abs(counts - 400) <= 100), counts)

    def test_equal_compute_partial_batches_leakage_and_repeatability(self):
        x, y, ids = self.fixture()
        results = [train_stream(x, y, ids, 211, 0.1, arm) for arm in
                   ('reservoir', 'fifo', 'current_only')]
        # Five blocks of 18 -> each has one 16-row and one 2-row batch.
        for r in results:
            self.assertEqual(r['counts'], dict(fresh_observations=90, fresh_presentations=360, replay=360, presentations=720, updates=40))
            self.assertEqual(sum(r['replay_counts']), 360)
            self.assertEqual(r['fresh_counts'], [1] * 90)
            self.assertTrue(set(r['replayed_row_ids']) <= set(ids))
            self.assertEqual(r['boundaries'][0]['empty_buffer_fallback_updates'], 4 if r['arm'] != 'current_only' else 8)
        again = train_stream(x, y, ids, 211, 0.1, 'reservoir')
        self.assertEqual(results[0]['boundaries'], again['boundaries'])
        for a, b in zip(results[0]['snapshots'][-1], again['snapshots'][-1]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(len({r['initial_sha256'] for r in results}), 1)
        self.assertEqual(len({str(r['label_order']) for r in results}), 1)

    def test_metrics_and_selection(self):
        y = np.repeat(np.arange(10), 2)
        guess = y.copy(); guess[0] = 1
        self.assertAlmostEqual(score(guess, y)['macro_accuracy'], .95)
        a = np.array([[.8, .1, .1, .1, .1], [.7, .9, .1, .1, .1],
                      [.9, .8, .8, .1, .1], [.7, .7, .9, .9, .1],
                      [.5, .6, .7, 1., .9]])
        self.assertAlmostEqual(forgetting(a), (.4+.3+.2-.1)/4)
        self.assertEqual(choose_rate({.3:[.7,.9]*3, .1:[.8,.8]*3, .03:[.6,.6]*3}), .1)
        np.testing.assert_allclose(paired_summary(np.ones(10)*.04)['interval_95'], [.04, .04], rtol=0, atol=1e-15)
        with self.assertRaises(ValueError):
            score([0], [0])
        with self.assertRaises(ValueError):
            train_stream(*self.fixture(), 1, float('nan'), 'reservoir')


if __name__ == '__main__':
    unittest.main()
