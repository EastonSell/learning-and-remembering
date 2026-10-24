"""Portable regression checks for saved scalars only; never run a model/bootstrap.

Run from the repository root:
    python -B -S -m unittest discover -s tests -p test_replay_decomposition_arithmetic.py -v

For an external, read-only evidence directory, set REPLAY_DECOMPOSITION_STUDY.
These checks supplement, and do not replace, the frozen study's full verifier.
"""

import copy
import importlib.util
import math
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "replay_arithmetic", ROOT / "scripts/check_replay_decomposition.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
STUDY = Path(os.environ.get(
    "REPLAY_DECOMPOSITION_STUDY",
    ROOT / "studies/replay-update-decomposition",
))


class ArithmeticContracts(unittest.TestCase):
    def test_factorial_interaction_and_conditional_effects(self):
        # Hand calculation: W=3, L=2, I=4; total=9, not W+L=5.
        got = audit.factorial({"00": 1.0, "01": 3.0, "10": 4.0, "11": 10.0})
        self.assertEqual(got, {
            "weight_at_original": 3.0,
            "location_at_deployed": 2.0,
            "interaction": 4.0,
            "total_sequential_minus_joint": 9.0,
            "weight_at_updated": 7.0,
            "location_at_unit": 6.0,
            "reconciliation_residual": 0.0,
        })

    def test_negative_mean_is_not_every_state_or_every_seed(self):
        # Four mathematical blocks per seed, not empirical observations.
        levels = {2005: -10.0, 2006: 1.0, 2007: 1.0, 2008: 1.0}
        records = [(s, b, levels[s]) for s in audit.SEEDS for b in audit.BLOCKS]
        got = audit.cluster_summary(records)
        self.assertEqual(got["mean"], -1.75)
        self.assertEqual(got["state_sign_counts"], {"negative": 4, "zero": 0, "positive": 12})
        self.assertEqual(got["seed_sign_counts"], {"negative": 1, "zero": 0, "positive": 3})
        self.assertEqual((got["cluster_count"], got["state_count"]), (4, 16))

    def test_zero_is_not_a_benefit_or_harm(self):
        self.assertEqual(audit.sign_counts([-1.0, -0.0, 0.0, 2.0]),
                         {"negative": 1, "zero": 2, "positive": 1})

    def test_duplicate_or_missing_blocks_cannot_be_pooled_into_clusters(self):
        records = [(s, b, float(s)) for s in audit.SEEDS for b in audit.BLOCKS]
        duplicate_replacement = records.copy()
        duplicate_replacement[1] = duplicate_replacement[0]
        for changed in (records[:-1], records + [records[0]], duplicate_replacement,
                        [(9999, 0, 1.0)] + records[1:]):
            with self.subTest(records=changed):
                with self.assertRaisesRegex(ValueError, "seed/block"):
                    audit.cluster_summary(changed)

    def test_nonfinite_values_do_not_silently_disappear_from_sign_counts(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    audit.sign_counts([value])


class PublishedArithmetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.saved = audit.load_evidence(STUDY)
        cls.actual = audit.recompute(cls.rows)

    def test_all_factorial_families_and_mir_controls_match_saved_summary(self):
        # Both panels, all four contrasts, ten non-residual factorial outcomes,
        # plus all 24 original/fresh-control MIR effects. No family selection.
        result = audit.compare_saved(self.actual, self.saved)
        self.assertEqual(result["checked_summaries"], 104)
        self.assertEqual((result["clusters"], result["states"]), (4, 16))
        self.assertFalse(result["bootstrap_recomputed"])
        self.assertFalse(result["model_code_imported"])

    def test_old_joint_average_does_not_erase_seven_adverse_states(self):
        got = self.actual["old.mir_minus_highloss.cell00"]
        self.assertAlmostEqual(got["mean"], -0.02506858497444485, places=12)
        self.assertEqual(got["state_sign_counts"], {"negative": 9, "zero": 0, "positive": 7})
        self.assertEqual(got["seed_sign_counts"], {"negative": 3, "zero": 0, "positive": 1})
        self.assertGreater(got["seed_means"]["2006"], 0.0)

    def test_favorable_isolated_effect_means_are_not_universal(self):
        expected = {
            "weight_at_original": (-0.1069226502970601, 10, 6),
            "location_at_deployed": (-0.04496275775500917, 9, 7),
        }
        for component, (mean, negative, positive) in expected.items():
            with self.subTest(component=component):
                got = self.actual["old.mir_minus_highloss." + component]
                self.assertAlmostEqual(got["mean"], mean, places=12)
                self.assertEqual(got["state_sign_counts"],
                                 {"negative": negative, "zero": 0, "positive": positive})
                self.assertEqual(got["seed_sign_counts"], {"negative": 3, "zero": 0, "positive": 1})

    def test_positive_interaction_seed_means_do_not_mean_every_state(self):
        got = self.actual["old.mir_minus_highloss.interaction"]
        self.assertAlmostEqual(got["mean"], 0.5170774349264109, places=12)
        self.assertEqual(got["seed_sign_counts"], {"negative": 0, "zero": 0, "positive": 4})
        self.assertEqual(got["state_sign_counts"], {"negative": 3, "zero": 0, "positive": 13})

    def test_absolute_old_damage_remains_despite_relative_mean_advantage(self):
        got = self.actual["cell00.mir.old_change_from_original"]
        self.assertAlmostEqual(got["mean"], 0.9055192529057605, places=12)
        self.assertEqual(got["state_sign_counts"], {"negative": 0, "zero": 0, "positive": 16})
        self.assertEqual(got["seed_sign_counts"], {"negative": 0, "zero": 0, "positive": 4})

    def test_fresh_control_scale_changes_the_new_panel_mean_conclusion(self):
        deployed = self.actual["cell00.mir.new_change_from_fresh_deployed"]
        unit = self.actual["cell00.mir.new_change_from_fresh_unit"]
        self.assertAlmostEqual(deployed["mean"], -0.057774, places=6)
        self.assertAlmostEqual(unit["mean"], 0.335974, places=6)
        self.assertEqual(unit["state_sign_counts"], {"negative": 4, "zero": 0, "positive": 12})
        relative = self.actual["new.mir_minus_highloss.cell00"]
        self.assertLess(relative["mean"], 0.0)
        self.assertEqual(relative["state_sign_counts"], {"negative": 5, "zero": 0, "positive": 11})

    def test_saved_interval_excludes_zero_but_not_engineering_margin(self):
        # This reads the recorded interval; it does not reproduce its bootstrap.
        got = self.saved["old.mir_minus_highloss.cell00"]
        low, high = got["ci95"]
        self.assertLess(low, high)
        self.assertLess(high, 0.0)
        self.assertGreater(high, -0.005)
        self.assertLess(got["mean"], -0.005)

    def test_mutated_interaction_mean_is_rejected(self):
        changed = copy.deepcopy(self.saved)
        changed["old.mir_minus_highloss.interaction"]["mean"] *= -1
        with self.assertRaisesRegex(ValueError, "interaction.*mean"):
            audit.compare_saved(self.actual, changed)

    def test_mean_sign_cannot_replace_state_sign_counts(self):
        changed = copy.deepcopy(self.saved)
        changed["old.mir_minus_highloss.cell00"]["state_sign_counts"] = {
            "negative": 16, "zero": 0, "positive": 0,
        }
        with self.assertRaisesRegex(ValueError, "cell00.*state_sign_counts"):
            audit.compare_saved(self.actual, changed)

    def test_states_cannot_be_relabelled_as_independent_clusters(self):
        changed = copy.deepcopy(self.saved)
        changed["old.mir_minus_highloss.cell00"]["cluster_count"] = 16
        with self.assertRaisesRegex(ValueError, "cluster_count"):
            audit.compare_saved(self.actual, changed)

    def test_raw_branch_loss_change_is_detected_without_using_cached_metrics(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["cells"]["00"]["branches"]["mir"]["old_loss"] += 0.01
        # Cached contrasts, decompositions and row metrics remain untouched.
        with self.assertRaisesRegex(ValueError, "mean|sign_counts"):
            audit.compare_saved(audit.recompute(changed), self.saved)

    def test_cached_state_metrics_are_not_used_as_an_arithmetic_oracle(self):
        changed = copy.deepcopy(self.rows)
        for row in changed:
            row.pop("metrics")
            row.pop("decompositions")
            for cell in row["cells"].values():
                cell.pop("contrasts")
        self.assertEqual(audit.recompute(changed), self.actual)


if __name__ == "__main__":
    unittest.main()
