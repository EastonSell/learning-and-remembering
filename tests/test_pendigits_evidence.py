"""Corruption regressions for the read-only PenDigits evidence checker."""

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check_pendigits_evidence.py"
STUDY = ROOT / "studies/pendigits-recurrent-replay"
SPEC = importlib.util.spec_from_file_location("pendigits_check", CHECKER)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class SupportGateTests(unittest.TestCase):
    def summary(self, differences, acquisition, interval, support):
        # Synthetic arithmetic fixtures, not extra experimental outcomes.
        by = {}
        for seed, difference in zip(range(31001, 31011), differences):
            by[seed, "uniform"] = {"final_macro": 0., "mean_acquisition": 0.5}
            by[seed, "mir"] = {"final_macro": difference,
                              "mean_acquisition": 0.5 + acquisition}
            by[seed, "high_loss"] = {"final_macro": 0.2, "mean_acquisition": 0.5}
        mean = sum(differences) / 10
        saved = {
            "mir_minus_uniform": differences, "paired_seed_bootstrap_95": interval,
            "mean_difference": mean, "mean_acquisition_difference": acquisition,
            "mean_final_macro": {"uniform": 0., "mir": mean, "high_loss": 0.2},
            "support": support, "test_rows": 3498,
            "wall_seconds": 1., "peak_rss_bytes": 1024,
        }
        return audit.check_summary(saved, by)

    def test_all_three_gates_can_pass(self):
        result = self.summary([0.02] * 10, 0., [0.02, 0.02], True)
        self.assertTrue(result["support"])

    def test_positive_interval_alone_does_not_satisfy_minimum_effect(self):
        result = self.summary([0.005] * 10, 0., [0.005, 0.005], False)
        self.assertFalse(result["support"])

    def test_interval_touching_zero_is_not_strictly_positive(self):
        # Nine zero differences and one +.20 give mean +.02. Resampling ten
        # pairs yields the recorded-RNG percentile interval [0, .06].
        result = self.summary([0.] * 9 + [0.2], 0., [0., 0.06], False)
        self.assertFalse(result["support"])

    def test_primary_improvement_cannot_override_acquisition_failure(self):
        result = self.summary([0.02] * 10, -0.04, [0.02, 0.02], False)
        self.assertFalse(result["support"])


class PenDigitsEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.study = Path(self.temp.name) / "study"
        shutil.copytree(STUDY, self.study)

    def run_check(self, optimized=False):
        command = [sys.executable, "-B"]
        if optimized:
            command.append("-O")
        command.extend([str(CHECKER), "--study", str(self.study)])
        return subprocess.run(command, capture_output=True, text=True,
                              env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})

    def refresh_hash(self, name):
        # Semantic tests deliberately refresh one local catalog entry so they
        # cannot pass merely because the generic hash check catches a mutation.
        path = self.study / "compact-evidence-manifest.json"
        catalog = json.loads(path.read_text())
        catalog[name] = hashlib.sha256((self.study / name).read_bytes()).hexdigest()
        path.write_text(json.dumps(catalog))

    def mutate_json(self, name, change):
        path = self.study / name
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))
        self.refresh_hash(name)

    def assert_rejected(self, clue, optimized=False):
        result = self.run_check(optimized)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "FAIL")
        self.assertIn(clue, payload["error"])
        self.assertEqual(result.stderr, "")

    def test_published_evidence_passes_without_modification_even_optimized(self):
        def snapshot():
            return {str(p.relative_to(self.study)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in self.study.rglob("*") if p.is_file()}
        before = snapshot()
        for optimized in (False, True):
            with self.subTest(optimized=optimized):
                result = self.run_check(optimized)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["status"], "PASS")
                self.assertEqual((payload["cases"], payload["boundaries"]), (30, 300))
                self.assertEqual(payload["csv_rows"], 30)
                self.assertAlmostEqual(payload["mir_minus_uniform_pp"], -1.4643304569695348)
                self.assertAlmostEqual(payload["acquisition_difference_pp"], -7.884569616837256)
                self.assertEqual(payload["ci95_pp"], [-3.160106492331734, 0.33135525755903633])
                self.assertFalse(payload["support"])
                self.assertEqual(payload["support_components"], {
                    "mean_gain_at_least_one_point": False,
                    "interval_strictly_above_zero": False,
                    "acquisition_drop_at_most_two_points": False,
                })
                self.assertTrue(payload["bootstrap_recomputed"])
                self.assertFalse(payload["model_code_imported"])
        self.assertEqual(before, snapshot())

    def test_corrupted_hash_is_rejected_with_optimization_on_or_off(self):
        with (self.study / "REPORT.md").open("a") as stream:
            stream.write("\ncorrupted evidence\n")
        for optimized in (False, True):
            with self.subTest(optimized=optimized):
                self.assert_rejected("REPORT.md", optimized)

    def test_omitting_an_entry_does_not_shrink_the_integrity_check(self):
        path = self.study / "compact-evidence-manifest.json"
        catalog = json.loads(path.read_text())
        del catalog["REPORT.md"]
        path.write_text(json.dumps(catalog))
        self.assert_rejected("catalog inventory")

    def test_unexpected_catalog_path_is_rejected(self):
        path = self.study / "compact-evidence-manifest.json"
        catalog = json.loads(path.read_text())
        catalog["../../unrelated"] = "0" * 64
        path.write_text(json.dumps(catalog))
        self.assert_rejected("catalog inventory")

    def test_missing_file_is_a_structured_failure(self):
        (self.study / "results/COMPLETE.json").unlink()
        self.assert_rejected("COMPLETE.json")

    def test_symbolic_link_cannot_substitute_for_evidence(self):
        path = self.study / "REPORT.md"
        other = self.study.parent / "outside-report.md"
        path.rename(other)
        path.symlink_to(other)
        self.assert_rejected("symlink")

    def test_incomplete_marker_cannot_coexist_with_a_passing_receipt(self):
        (self.study / "results/INCOMPLETE").write_text("partial execution")
        self.assert_rejected("INCOMPLETE")

    def test_completion_count_must_match_full_case_coverage(self):
        self.mutate_json("results/COMPLETE.json", lambda x: x.update(cases=29))
        self.assert_rejected("completion cases")

    def test_completion_checkpoint_count_is_not_inference_verification(self):
        self.mutate_json("results/COMPLETE.json", lambda x: x.update(checkpoints=299))
        self.assert_rejected("completion checkpoints")

    def test_completion_source_must_match_frozen_source(self):
        self.mutate_json("results/COMPLETE.json", lambda x: x.update(source_sha256="0" * 64))
        self.assert_rejected("completion source")

    def test_duplicate_case_cannot_replace_a_missing_case(self):
        self.mutate_json("results/metrics.json", lambda x: x.__setitem__(1, x[0]))
        self.assert_rejected("case identity")

    def test_missing_case_is_rejected(self):
        self.mutate_json("results/metrics.json", lambda x: x.pop())
        self.assert_rejected("30 cases")

    def test_phase_number_is_not_a_boolean(self):
        self.mutate_json("results/metrics.json", lambda x: x[0]["boundaries"][0].update(phase=False))
        self.assert_rejected("phase")

    def test_recurrent_pair_order_is_checked(self):
        self.mutate_json("results/metrics.json", lambda x: x[0]["boundaries"][5].update(current_pair=[0, 1]))
        self.assert_rejected("current_pair")

    def test_old_seen_class_accuracy_is_recomputed(self):
        self.mutate_json("results/metrics.json", lambda x: x[0]["boundaries"][1].update(old_accuracy=0.99))
        self.assert_rejected("old_accuracy")

    def test_first_phase_has_no_old_class_average(self):
        self.mutate_json("results/metrics.json", lambda x: x[0]["boundaries"][0].update(old_accuracy=0.0))
        self.assert_rejected("old_accuracy")

    def test_out_of_range_recall_is_rejected(self):
        self.mutate_json("results/metrics.json", lambda x: x[0]["boundaries"][0]["recalls"].__setitem__(0, 1.01))
        self.assert_rejected("recalls")

    def test_nonfinite_json_is_rejected_even_with_updated_hash(self):
        self.mutate_json("results/metrics.json", lambda x: x[0]["boundaries"][0].update(ce=float("nan")))
        self.assert_rejected("non-finite")

    def test_overflowing_json_number_is_rejected_even_in_extra_metadata(self):
        path = self.study / "results/summary.json"
        original = path.read_text()
        for token in ("1e999", "-1e999"):
            with self.subTest(token=token):
                path.write_text('{"extra_metadata": ' + token + ',' + original[1:])
                self.refresh_hash("results/summary.json")
                self.assert_rejected("non-finite")

    def test_duplicate_json_keys_are_rejected(self):
        path = self.study / "results/summary.json"
        original = path.read_text()
        path.write_text('{"support": true,' + original[1:])
        self.refresh_hash("results/summary.json")
        self.assert_rejected("duplicate JSON key")

    def test_finite_cross_entropy_is_still_required_to_be_nonnegative(self):
        self.mutate_json("results/metrics.json", lambda x: x[0]["boundaries"][0].update(ce=-1.0))
        self.assert_rejected("ce")

    def test_macro_is_recomputed_from_recalls_not_trusted(self):
        self.mutate_json("results/metrics.json", lambda x: x[0]["boundaries"][0].update(macro=0.5))
        self.assert_rejected("macro")

    def test_acquisition_is_recomputed_from_the_current_pair(self):
        self.mutate_json("results/metrics.json", lambda x: x[0]["boundaries"][0].update(acquisition=0.5))
        self.assert_rejected("acquisition")

    def test_equal_but_wrong_accounting_cannot_pass(self):
        def change(rows):
            for row in rows:
                row["counts_per_arm"]["actual_steps"] = 761
        self.mutate_json("results/metrics.json", change)
        self.assert_rejected("counts_per_arm")

    def test_pairwise_difference_cannot_be_replaced_with_an_unpaired_value(self):
        self.mutate_json("results/summary.json", lambda x: x["mir_minus_uniform"].__setitem__(0, 0.0))
        self.assert_rejected("mir_minus_uniform")

    def test_bootstrap_interval_is_recomputed(self):
        self.mutate_json("results/summary.json", lambda x: x.update(paired_seed_bootstrap_95=[-0.01, 0.01]))
        self.assert_rejected("paired_seed_bootstrap_95")

    def test_a_favorable_support_label_cannot_replace_the_failed_gate(self):
        self.mutate_json("results/summary.json", lambda x: x.update(support=True))
        self.assert_rejected("support")

    def test_csv_must_agree_with_saved_json(self):
        path = self.study / "results/seed-results.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["final_macro_accuracy"] = "1.0"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.refresh_hash("results/seed-results.csv")
        self.assert_rejected("CSV final_macro_accuracy")

    def test_csv_duplicate_cannot_replace_an_adverse_case(self):
        path = self.study / "results/seed-results.csv"
        lines = path.read_text().splitlines()
        lines[2] = lines[1]
        path.write_text("\n".join(lines) + "\n")
        self.refresh_hash("results/seed-results.csv")
        self.assert_rejected("CSV case identity")


if __name__ == "__main__":
    unittest.main()
