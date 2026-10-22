"""Controls fail if role masks change capacity or semantic renaming changes behavior."""
import unittest
import numpy as np
from functional_circuits import role_controls as controls
from functional_circuits.data import generate
from functional_circuits.models import Model, train
from functional_circuits.metrics import choose_mapping


class RoleControlTests(unittest.TestCase):
    def test_partition_dose_capacity_and_nesting(self):
        previous = set()
        for swaps in range(4):
            m = controls.make_control(11, swaps)
            self.assertEqual(m.parameter_count, 132)
            self.assertEqual(sorted(np.concatenate(m.groups).tolist()), list(range(12)))
            moved = set(m.groups[0]) & set(range(6, 12))
            self.assertEqual(len(moved), swaps)
            self.assertTrue(previous <= moved)
            previous = moved
        baseline = Model('functional', 11)
        aligned = controls.make_control(11, 0)
        for key in baseline.params:
            np.testing.assert_array_equal(baseline.params[key], aligned.params[key])
        for seed, swaps in [(11, -1), (11, 4), (11, 1.5)]:
            with self.assertRaises(ValueError):
                controls.make_control(seed, swaps)

    def test_branch_renaming_is_function_and_intervention_invariant(self):
        data, donors = generate(100, 11), generate(100, 12)
        m = controls.make_control(11, 2)
        train(m, data, 12, epochs=3)
        original = {k: v.copy() for k, v in m.params.items()}
        renamed = controls.rename_branches(m)
        np.testing.assert_allclose(m.forward(data.x)[0], renamed.forward(data.x)[0], atol=1e-14)
        for branch in (0, 1):
            np.testing.assert_allclose(m.patch(data.x, donors.x, [branch]),
                                       renamed.patch(data.x, donors.x, [1-branch]), atol=1e-14)
        for key in original:
            np.testing.assert_array_equal(original[key], m.params[key])
        mapping, metadata = choose_mapping(m, data, donors)
        self.assertEqual(metadata['mapping_candidates'], 2)

    def test_runner_is_repeatable_and_protects_outputs(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            root = Path(folder)
            args = ['--seeds', '11', '--epochs', '2', '--train-size', '64',
                    '--test-size', '64', '--validation-size', '32']
            controls.main(['--output', str(root/'a'), *args])
            controls.main(['--output', str(root/'b'), *args])
            a = json.loads((root/'a'/'metrics.json').read_text())
            b = json.loads((root/'b'/'metrics.json').read_text())
            self.assertEqual(a, b)
            self.assertEqual(len(a), 12)
            self.assertEqual({r['parameters'] for r in a}, {132})
            with self.assertRaises(SystemExit):
                controls.main(['--output', str(root/'a'), *args])

    def test_absent_context_rejected_before_output_creation(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            out = Path(folder)/'invalid'
            with self.assertRaises(SystemExit):
                controls.main(['--output', str(out), '--seeds', '11',
                               '--test-size', '1', '--epochs', '1', '--train-size', '8'])
            self.assertFalse(out.exists())
