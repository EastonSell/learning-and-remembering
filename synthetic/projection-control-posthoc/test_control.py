import io
import unittest
from contextlib import redirect_stdout
import control

class ProjectionTests(unittest.TestCase):
    def test_non_axis_aligned_projection(self):
        d = control.project((2., 1.), [(1., 1.)])
        self.assertAlmostEqual(d[0], .5)
        self.assertAlmostEqual(d[1], -.5)

    def test_dependent_and_full_rank_gradients(self):
        d = control.project((2., 3.), [(1., 1.), (2., 2.), (1., -1.)])
        self.assertAlmostEqual(sum(x*x for x in d), 0.)

    def test_empty_span_and_zero_gradient(self):
        self.assertEqual(control.project((2., 3.), [(0., 0.)]), (2., 3.))
        self.assertEqual(control.project((2., 3.), []), (2., 3.))

    def test_near_dependency_threshold_and_row_order(self):
        from math import sqrt
        h = (0., 1.)
        below = (sqrt(1 - .099**2), .099)
        above = (sqrt(1 - .101**2), .101)
        self.assertAlmostEqual(control.project(h, [(1., 0.), below], .1)[1], 1.)
        self.assertAlmostEqual(control.project(h, [(1., 0.), above], .1)[1], 0.)
        forward = control.project(h, [(1., 0.), below], .1)
        reverse = control.project(h, [below, (1., 0.)], .1)
        self.assertGreater(abs(forward[0] - reverse[0]), .09)

    def test_invalid_dimension_and_nonfinite_values(self):
        for h, gradients in [((1., 2.), [(1.,)]), ((float('nan'),), []),
                             ((1.,), [(float('inf'),)]), ((), [])]:
            with self.subTest(h=h, gradients=gradients):
                with self.assertRaises(ValueError): control.project(h, gradients)
        for tol in [-1., 1., float('nan')]:
            with self.assertRaises(ValueError): control.project((1.,), [], tol)

    def test_higher_dimensional_orthogonality_and_idempotence(self):
        from math import fsum
        rows = [(1., 2., -1., 0.), (0., 1., 3., 1.), (2., 4., -2., 0.)]
        result = control.project((3., -1., 4., 2.), rows)
        for row in rows:
            self.assertAlmostEqual(fsum(a*b for a, b in zip(row, result)), 0., places=12)
        for a, b in zip(result, control.project(result, rows)):
            self.assertAlmostEqual(a, b, places=12)

    def test_main_reports_four_actual_assertions(self):
        out = io.StringIO()
        with redirect_stdout(out): status = control.main()
        self.assertEqual(status, 0)
        self.assertEqual(out.getvalue().count('PASS '), 4)
        self.assertIn('4/4 deterministic assertions passed', out.getvalue())

    def test_reports_all_failures_and_full_denominator(self):
        out = io.StringIO()
        with redirect_stdout(out):
            status = control.report([('a', 1., 0.), ('b', 0., 0.), ('c', 2., 0.)])
        self.assertEqual(status, 1)
        self.assertIn('FAIL a', out.getvalue())
        self.assertIn('FAIL c', out.getvalue())
        self.assertIn('1/3', out.getvalue())

if __name__ == '__main__': unittest.main()
