"""Tests for the illustrative, standard-library replay lesson only."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

PATH = Path(__file__).with_name('replay_demo.py')


class ReplayLessonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = importlib.util.spec_from_file_location('replay_lesson', PATH)

    def load_demo(self):
        self.assertTrue(PATH.exists(), 'The offline replay lesson is not implemented yet')
        module = importlib.util.module_from_spec(self.spec)
        self.spec.loader.exec_module(module)
        return module

    def test_two_tasks_have_one_shared_exact_solution(self):
        demo = self.load_demo()
        for example in (demo.TASK_A, demo.TASK_B):
            self.assertEqual(demo.loss((1.0, -1.0), example), 0.0)

    def test_replay_reduces_old_task_loss_with_matched_budgets(self):
        demo = self.load_demo()
        result = demo.run()
        self.assertLess(result['after_a']['a_loss'], 1e-9)
        self.assertGreater(result['current_only']['a_loss'], 0.12)
        self.assertLess(result['current_only']['b_loss'], 1e-20)
        self.assertLess(result['replay']['a_loss'], 0.001)
        self.assertLess(result['replay']['b_loss'], 0.001)
        self.assertEqual(result['current_only']['updates'], 120)
        self.assertEqual(result['replay']['updates'], 120)
        self.assertEqual(result['current_only']['presentations'], 240)
        self.assertEqual(result['replay']['presentations'], 240)
        self.assertEqual(len(result['replay']['weights']), 2)
        self.assertEqual(result, demo.run())

    def test_step_averages_examples_and_does_not_mutate_inputs(self):
        demo = self.load_demo()
        weights = (1.0, 0.0)
        self.assertEqual(demo.step(weights, (demo.TASK_B, demo.TASK_B)), (0.75, -0.25))
        self.assertEqual(weights, (1.0, 0.0))

    def test_current_loss_and_predicted_harm_pick_different_items(self):
        demo = self.load_demo()
        result = demo.selection_example()
        self.assertEqual(result['high_loss_choice'], 'hard_now')
        self.assertEqual(result['predicted_harm_choice'], 'easy_but_threatened')
        by_name = {row['name']: row for row in result['candidates']}
        self.assertAlmostEqual(by_name['hard_now']['loss_now'], 0.5)
        self.assertLess(by_name['hard_now']['loss_increase'], 0)
        self.assertAlmostEqual(by_name['easy_but_threatened']['loss_now'], 0)
        self.assertAlmostEqual(by_name['easy_but_threatened']['loss_increase'], 0.125)
        self.assertEqual(result['uniform_probabilities'], [1/3, 1/3, 1/3])
        self.assertEqual(result['weights_before'], [1.0, 0.0])
        self.assertEqual(result, demo.selection_example())

    def test_cli_explicitly_labels_illustration(self):
        demo = self.load_demo()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            demo.main()
        self.assertIn('ILLUSTRATION', output.getvalue())
        self.assertIn('not benchmark evidence', output.getvalue())


if __name__ == '__main__':
    unittest.main()
