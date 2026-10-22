"""Synthetic retention and adversarial verification tests: no research training."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("replaystudy.evidence"),
                             "independent evidence module must exist")
        self.e = importlib.import_module("replaystudy.evidence")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.provenance = {}
        for key in ("source", "tests", "protocol", "data", "indices"):
            path = self.root / (key + ".txt")
            path.write_text("synthetic " + key + "\n", encoding="utf-8")
            self.provenance[key] = path
        self.expected = [self.identity(1), self.identity(2)]

    def identity(self, seed):
        return {"case_id": self.e.make_case_id(seed, "mir", 0.1),
                "seed": seed, "arm": "mir", "eta": 0.1}

    def bundle(self, name="run", **kwargs):
        return self.e.Bundle(self.root / name, stage="software",
                             expected_cases=self.expected,
                             provenance=self.provenance, **kwargs)

    def record(self, bundle, seed=1):
        record = dict(self.identity(seed), status="ok")
        record["checkpoints"] = []
        for block in range(5):
            rel = f"checkpoints/{record['case_id']}/block_{block}.bin"
            path = bundle.path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic {seed} block {block}".encode())
            record["checkpoints"].append({"block": block, "path": rel,
                                           "sha256": self.e.sha256_file(path)})
        record["metrics"] = {"validation_recall": [0.5] * 10}
        return record

    def research_record(self, bundle, identity):
        import numpy as np
        row = dict(identity, status="ok", checkpoints=[])
        case_path = bundle.path / identity["case_id"]
        (case_path / "training").mkdir(parents=True)
        for block in range(5):
            path = case_path / "training" / f"block-{block}.npz"
            np.savez(path, p0=np.zeros((196, 32), np.float32), p1=np.zeros(32, np.float32),
                     p2=np.zeros((32, 10), np.float32), p3=np.zeros(10, np.float32))
            row["checkpoints"].append({"block": block, "path": path.relative_to(bundle.path).as_posix(),
                                       "sha256": self.e.sha256_file(path)})
        np.save(case_path / "stream-indices.npy", np.arange(10000, dtype=np.uint64), allow_pickle=False)
        row["stream_sha256"] = self.e.sha256_file(case_path / "stream-indices.npy")
        for key in ("candidate_trace_sha256", "reservoir_trace_sha256", "quota_trace_sha256"):
            row[key] = hashlib.sha256(f"synthetic {identity['seed']} {key}".encode()).hexdigest()
        row["class_pairs"] = np.arange(10).reshape(5, 2).tolist()
        extra = identity["arm"] == "uniform-extra-replay"
        row["ledger"] = {
            "committed_base_updates": 2500, "fresh_presentations": 40000,
            "current_repeat_presentations": 40, "historical_presentations": 24960,
            "unique_observations": 10000, "trainable_scalars": 6634,
            "trainable_raw_bytes": 26536, "reservoir_raw_bytes": 158600,
            "virtual_gradients": 0 if extra else 2500,
            "operations_allowed": 6200562008, "operations_used": 6200428622 if extra else 6200562008,
            "residual_operations": 133386 if extra else 0,
            "extra_updates": 15853 if extra else 0, "candidate_draws": 124648,
            "candidate_forward_examples": 0 if extra else 249296,
            "base_selected_items": 25000, "reservoir_counter": 10000,
            "score_count": 0 if extra else 124648,
        }
        with (case_path / "mechanism-audit.jsonl").open("w") as handle:
            for update in range(525, 2501, 25):
                handle.write(json.dumps({"update": update,
                    "candidate_rank_vs_exhaustive_harm": 1., "selected_mean_harm": 0.,
                    "eligible_mean_harm": 0., "mir_highloss_overlap": .5,
                    "selected_minus_shuffled_old_loss": 0., "audit_operations": 100,
                    "wall_seconds": 0., "cpu_seconds": 0.}) + "\n")
        if bundle._stage == "development":
            row["metrics"] = {"validation_matrix": [[0.5] * 5] * 5,
                              "validation_recall": [0.5] * 10}
        else:
            row["metrics"] = {"test_matrix": [[0.5] * 5] * 5,
                              "full_test_matrix": [[0.5] * 5] * 5,
                              "test_recall": [0.5] * 10, "full_test_recall": [0.5] * 10}
        return row

    def research_bundle(self, name="research"):
        cases = self.e.expected_cases("development")
        bundle = self.e.Bundle(self.root / name, "development", cases, self.provenance)
        rows = [self.research_record(bundle, identity) for identity in cases]
        self.rewrite_records(bundle, rows)
        return bundle, rows

    def complete(self, name="run", **kwargs):
        bundle = self.bundle(name, **kwargs)
        with bundle:
            for seed in (1, 2):
                bundle.append_case(self.record(bundle, seed))
            bundle.complete()
        return bundle

    def rewrite_records(self, bundle, records):
        (bundle.path / "cases.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    def test_expected_development_and_evaluation_case_grids(self):
        dev = self.e.expected_cases("development")
        self.assertEqual(len(dev), 30)
        self.assertEqual({x["seed"] for x in dev}, {1001, 1002})
        self.assertEqual({x["eta"] for x in dev}, {0.03, 0.1, 0.3})
        ev = self.e.expected_cases("evaluation", diagnostic_eta=0.1, extra_eta=0.03)
        self.assertEqual(len(ev), 100)
        self.assertEqual({x["seed"] for x in ev}, set(range(2001, 2021)))
        self.assertEqual(len({x["case_id"] for x in ev}), 100)
        with self.assertRaises(ValueError):
            self.e.expected_cases("evaluation")

    def test_existing_output_is_rejected_without_modification(self):
        path = self.root / "run"
        path.mkdir()
        sentinel = path / "do_not_touch"
        sentinel.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            self.bundle()
        self.assertEqual(sentinel.read_bytes(), b"original")
        self.assertEqual(list(path.iterdir()), [sentinel])

    def test_empty_expected_cases_rejected(self):
        with self.assertRaises(ValueError):
            self.e.Bundle(self.root / "empty", stage="software", expected_cases=[],
                          provenance=self.provenance)

    def test_running_marker_precedes_cases_and_snapshots_provenance(self):
        bundle = self.bundle()
        self.assertTrue((bundle.path / "RUNNING").is_file())
        self.assertEqual((bundle.path / "cases.jsonl").read_bytes(), b"")
        manifest = json.loads((bundle.path / "manifest.json").read_text())
        self.assertEqual(set(manifest["provenance"]), set(self.provenance))
        for entries in manifest["provenance"].values():
            for entry in entries:
                self.assertEqual(self.e.sha256_file(bundle.path / entry["path"]), entry["sha256"])
        bundle.close()

    def test_interrupt_before_first_case_is_retained(self):
        bundle = self.bundle()
        with self.assertRaises(KeyboardInterrupt):
            with bundle:
                raise KeyboardInterrupt("injected before first case")
        self.assertTrue((bundle.path / "INTERRUPTED").exists())
        self.assertTrue((bundle.path / "RUNNING").exists())
        self.assertFalse((bundle.path / "COMPLETE").exists())
        self.assertEqual((bundle.path / "cases.jsonl").read_bytes(), b"")
        with self.assertRaises(self.e.VerificationError):
            self.e.verify_bundle(bundle.path)

    def test_interrupt_after_first_case_preserves_exact_record(self):
        bundle = self.bundle()
        record = self.record(bundle)
        with self.assertRaises(KeyboardInterrupt):
            with bundle:
                bundle.append_case(record)
                raise KeyboardInterrupt("injected after one case")
        self.assertEqual(json.loads((bundle.path / "cases.jsonl").read_text()), record)
        self.assertTrue((bundle.path / "INTERRUPTED").exists())
        with self.assertRaises(self.e.VerificationError):
            self.e.verify_bundle(bundle.path)

    def test_failure_before_first_case_is_retained(self):
        bundle = self.bundle()
        with self.assertRaises(RuntimeError):
            with bundle:
                raise RuntimeError("injected before first case")
        self.assertIn("injected before first case", (bundle.path / "FAILED").read_text())
        self.assertFalse((bundle.path / "COMPLETE").exists())

    def test_failed_case_is_written_before_raising(self):
        bundle = self.bundle()
        record = dict(self.identity(1), status="failed", error="diverged", predictions=[7, 2])
        with self.assertRaises(self.e.EvidenceError):
            with bundle:
                bundle.append_case(record)
        self.assertEqual(json.loads((bundle.path / "cases.jsonl").read_text()), record)
        self.assertTrue((bundle.path / "FAILED").exists())

    def test_wrong_prediction_is_retained_before_assertion(self):
        bundle = self.bundle()
        record = self.record(bundle)
        record["predictions"] = [0, 0]
        record["prediction_checks"] = {"fixture_prediction_matches": False}
        with self.assertRaises(self.e.EvidenceError):
            with bundle:
                bundle.append_case(record)
        self.assertEqual(json.loads((bundle.path / "cases.jsonl").read_text()), record)
        self.assertFalse((bundle.path / "COMPLETE").exists())

    def test_valid_complete_bundle_and_independent_verification(self):
        bundle = self.complete()
        self.assertTrue((bundle.path / "COMPLETE").exists())
        self.assertFalse((bundle.path / "RUNNING").exists())
        result = self.e.verify_bundle(bundle.path)
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["stage"], "software")
        self.assertTrue(result["complete"])
        with self.assertRaises(self.e.EvidenceError):
            bundle.append_case(self.record(bundle))

    def test_missing_or_empty_coverage_cannot_complete(self):
        for name, n in (("empty", 0), ("missing", 1)):
            with self.subTest(name=name):
                bundle = self.bundle(name)
                with bundle:
                    if n:
                        bundle.append_case(self.record(bundle))
                    with self.assertRaises(self.e.VerificationError):
                        bundle.complete()
                self.assertFalse((bundle.path / "COMPLETE").exists())

    def test_duplicate_and_unexpected_identity_are_retained_and_rejected(self):
        for name in ("duplicate", "unexpected", "wrong_config"):
            with self.subTest(name=name):
                bundle = self.bundle(name)
                first = self.record(bundle)
                with self.assertRaises(self.e.EvidenceError):
                    with bundle:
                        bundle.append_case(first)
                        second = self.record(bundle, 1 if name == "duplicate" else 2)
                        if name == "unexpected":
                            second["case_id"] = "not-declared"
                        if name == "wrong_config":
                            second["eta"] = 0.3
                        bundle.append_case(second)
                self.assertEqual(len((bundle.path / "cases.jsonl").read_text().splitlines()), 2)
                self.assertFalse((bundle.path / "COMPLETE").exists())

    def test_exactly_five_distinct_checkpoints_required(self):
        for name, alter in (("missing", lambda xs: xs[:-1]),
                            ("duplicate", lambda xs: xs[:4] + [xs[0]])):
            bundle = self.bundle(name)
            row = self.record(bundle)
            row["checkpoints"] = alter(row["checkpoints"])
            with self.assertRaises(self.e.EvidenceError):
                with bundle:
                    bundle.append_case(row)
            self.assertEqual(len((bundle.path / "cases.jsonl").read_text().splitlines()), 1)

    def test_checkpoint_content_is_rehashed(self):
        bundle = self.bundle()
        row = self.record(bundle)
        (bundle.path / row["checkpoints"][0]["path"]).write_bytes(b"tampered")
        with self.assertRaises(self.e.EvidenceError):
            with bundle:
                bundle.append_case(row)

    def test_nonfinite_and_wrong_shape_metrics_are_retained(self):
        for name, values in (("nan", [float("nan")] * 10), ("shape", [0.5] * 9)):
            bundle = self.bundle(name)
            row = self.record(bundle)
            row["metrics"]["validation_recall"] = values
            with self.assertRaises(self.e.EvidenceError):
                with bundle:
                    bundle.append_case(row)
            self.assertIn("validation_recall", (bundle.path / "cases.jsonl").read_text())

    def test_stage_specific_shape_schema_is_enforced(self):
        bundle = self.bundle(required_fields={"metrics.accuracy_matrix": [5, 5]})
        row = self.record(bundle)
        row["metrics"]["accuracy_matrix"] = [[0.5] * 5] * 4
        with self.assertRaises(self.e.EvidenceError):
            with bundle:
                bundle.append_case(row)

    def test_artifact_inventory_detects_any_modified_added_or_removed_file(self):
        for name in ("modified", "added", "removed"):
            bundle = self.complete(name)
            if name == "modified":
                (bundle.path / "cases.jsonl").write_text("{}\n")
            elif name == "added":
                (bundle.path / "unlisted.txt").write_text("extra")
            else:
                next((bundle.path / "checkpoints").rglob("*.bin")).unlink()
            with self.assertRaises(self.e.VerificationError):
                self.e.verify_bundle(bundle.path)

    def test_completed_bundle_provenance_mutation_detected(self):
        bundle = self.complete()
        manifest = json.loads((bundle.path / "manifest.json").read_text())
        entry = manifest["provenance"]["source"][0]
        (bundle.path / entry["path"]).write_bytes(b"changed source")
        with self.assertRaises(self.e.VerificationError):
            self.e.verify_bundle(bundle.path)

    def test_truncated_jsonl_rejected_even_with_complete_coverage(self):
        bundle = self.bundle()
        with bundle:
            for seed in (1, 2):
                bundle.append_case(self.record(bundle, seed))
            with (bundle.path / "cases.jsonl").open("ab") as handle:
                handle.write(b'{"partial":')
            with self.assertRaises(self.e.VerificationError):
                bundle.complete()

    def test_path_traversal_and_symlink_artifacts_rejected(self):
        bundle = self.bundle()
        row = self.record(bundle)
        row["checkpoints"][0]["path"] = "../source.txt"
        with self.assertRaises(self.e.EvidenceError):
            with bundle:
                bundle.append_case(row)
        bundle = self.complete("symlink")
        (bundle.path / "outside").symlink_to(self.root / "source.txt")
        with self.assertRaises(self.e.VerificationError):
            self.e.verify_bundle(bundle.path)

    def test_research_stages_reject_partial_expected_case_grid(self):
        expected = self.e.expected_cases("development")[:-1]
        with self.assertRaises(ValueError):
            self.e.Bundle(self.root / "baddev", stage="development",
                          expected_cases=expected, provenance=self.provenance)

    def test_evaluation_requires_lock_provenance_and_matrix_schema(self):
        expected = self.e.expected_cases("evaluation", diagnostic_eta=0.1, extra_eta=0.03)
        with self.assertRaises(ValueError):
            self.e.Bundle(self.root / "bad_evaluation", stage="evaluation",
                          expected_cases=expected, provenance=self.provenance)
        lock = self.root / "lock.json"
        lock.write_text("{}\n")
        provenance = dict(self.provenance, lock=lock)
        bundle = self.e.Bundle(self.root / "evaluation", stage="evaluation",
                               expected_cases=expected, provenance=provenance)
        schema = json.loads((bundle.path / "manifest.json").read_text())["required_fields"]
        self.assertEqual(schema["metrics.test_matrix"], [5, 5])
        self.assertEqual(schema["metrics.full_test_matrix"], [5, 5])
        bundle.close()


    def test_independent_verifier_rejects_corrupt_rows_without_writer_checks(self):
        mutations = {
            "duplicate": lambda rows: [rows[0], rows[0]],
            "unexpected": lambda rows: [dict(rows[0], case_id="unknown"), rows[1]],
            "config": lambda rows: [dict(rows[0], eta=0.3), rows[1]],
            "failed": lambda rows: [dict(rows[0], status="failed"), rows[1]],
            "missing": lambda rows: rows[:1],
            "prediction": lambda rows: [dict(rows[0], prediction_checks={"expected": False}), rows[1]],
            "shape": lambda rows: [dict(rows[0], metrics={"validation_recall": [0.5]}), rows[1]],
            "checkpoints": lambda rows: [dict(rows[0], checkpoints=rows[0]["checkpoints"][:4]), rows[1]],
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                bundle = self.bundle(name)
                rows = [self.record(bundle, seed) for seed in (1, 2)]
                self.rewrite_records(bundle, mutate(rows))
                with self.assertRaises(self.e.VerificationError):
                    self.e.verify_bundle(bundle.path, require_complete=False)
                bundle.close()

    def test_checkpoint_paths_cannot_be_reused_between_cases(self):
        bundle = self.bundle()
        rows = [self.record(bundle, seed) for seed in (1, 2)]
        rows[1]["checkpoints"] = rows[0]["checkpoints"]
        self.rewrite_records(bundle, rows)
        with self.assertRaises(self.e.VerificationError):
            self.e.verify_bundle(bundle.path, require_complete=False)
        bundle.close()

    def test_unlisted_checkpoint_file_invalidates_exact_checkpoint_coverage(self):
        bundle = self.bundle()
        rows = [self.record(bundle, seed) for seed in (1, 2)]
        (bundle.path / "checkpoints" / "unlisted.bin").write_bytes(b"unlisted checkpoint")
        self.rewrite_records(bundle, rows)
        with self.assertRaises(self.e.VerificationError):
            self.e.verify_bundle(bundle.path, require_complete=False)
        bundle.close()

    def test_source_directory_is_snapshotted_with_relative_paths(self):
        directory = self.root / "code"
        (directory / "nested").mkdir(parents=True)
        (directory / "a.py").write_text("x = 1\n")
        (directory / "nested" / "b.py").write_text("y = 2\n")
        self.provenance["source"] = directory
        bundle = self.complete()
        manifest = json.loads((bundle.path / "manifest.json").read_text())
        self.assertEqual({x["path"] for x in manifest["provenance"]["source"]},
                         {"provenance/source/0/a.py", "provenance/source/0/nested/b.py"})
        (directory / "a.py").write_text("changed after freeze\n")
        self.assertTrue(self.e.verify_bundle(bundle.path)["complete"])

    def test_each_append_calls_fsync_before_return(self):
        from unittest import mock
        bundle = self.bundle()
        real_fsync = self.e.os.fsync
        with mock.patch.object(self.e.os, "fsync", wraps=real_fsync) as synced:
            bundle.append_case(self.record(bundle))
            self.assertGreaterEqual(synced.call_count, 1)
        self.assertEqual(len((bundle.path / "cases.jsonl").read_text().splitlines()), 1)
        bundle.close()

    def test_falsified_scientific_prediction_is_a_retained_valid_result(self):
        bundle = self.bundle()
        with bundle:
            for seed in (1, 2):
                row = self.record(bundle, seed)
                row["scientific_prediction"] = {"mir_beats_highloss": False}
                bundle.append_case(row)
            bundle.complete()
        self.assertTrue(self.e.verify_bundle(bundle.path)["complete"])

    def test_metric_bounds_and_boolean_metrics_are_rejected(self):
        for name, values in (("out_of_range", [1.1] * 10), ("boolean", [True] * 10)):
            bundle = self.bundle(name)
            row = self.record(bundle)
            row["metrics"]["validation_recall"] = values
            self.rewrite_records(bundle, [row, self.record(bundle, 2)])
            with self.assertRaises(self.e.VerificationError):
                self.e.verify_bundle(bundle.path, require_complete=False)
            bundle.close()

    def test_malformed_checkpoint_path_raises_verification_error(self):
        bundle = self.bundle()
        rows = [self.record(bundle, seed) for seed in (1, 2)]
        rows[0]["checkpoints"][0]["path"] = ["not", "a", "path"]
        self.rewrite_records(bundle, rows)
        with self.assertRaises(self.e.VerificationError):
            self.e.verify_bundle(bundle.path, require_complete=False)
        bundle.close()

    def test_artifact_manifest_digest_is_bound_by_complete_marker(self):
        bundle = self.complete()
        inventory = bundle.path / "artifacts.json"
        inventory.write_text(inventory.read_text() + " ")
        with self.assertRaises(self.e.VerificationError):
            self.e.verify_bundle(bundle.path)

    def test_duplicate_json_keys_are_rejected(self):
        bundle = self.bundle()
        rows = [self.record(bundle, seed) for seed in (1, 2)]
        self.rewrite_records(bundle, rows)
        path = bundle.path / "cases.jsonl"
        path.write_text(path.read_text().replace('"status": "ok"', '"status": "failed", "status": "ok"', 1))
        with self.assertRaises(self.e.VerificationError):
            self.e.verify_bundle(bundle.path, require_complete=False)
        bundle.close()


    def test_runner_path_convention_and_positional_constructor(self):
        bundle = self.e.Bundle(self.root / "runner", "software", self.expected, self.provenance)
        with bundle:
            for seed in (1, 2):
                row = self.record(bundle, seed)
                for entry in row["checkpoints"]:
                    source = bundle.path / entry["path"]
                    relative = f"{row['case_id']}/training/block-{entry['block']}.npz"
                    destination = bundle.path / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source.rename(destination)
                    entry["path"] = relative
                trace = destination.parent / "selection-trace.jsonl"
                trace.write_text("synthetic trace\n")
                bundle.append_case(row)
            bundle.complete()
        self.assertTrue(self.e.verify_bundle(bundle.path)["complete"])


    def test_evaluation_lock_configuration_is_bound_independently(self):
        cases = self.e.expected_cases("evaluation", diagnostic_eta=0.1, extra_eta=0.03)
        lock_value = {"kind": "evaluation-lock-v1", "complete": True,
                      "no_prior_test_selection": True,
                      "rates": {"diagnostic_eta": 0.1, "extra_eta": 0.03},
                      "seeds": list(range(2001, 2021)), "arms": list(self.e.ARMS),
                      "architecture": [196, 32, 10],
                      "data_manifest_sha256": self.e.sha256_file(self.provenance["data"])}
        lock = self.root / "lock.json"
        lock.write_text(json.dumps(lock_value))
        bundle = self.e.Bundle(self.root / "evaluation", "evaluation", cases,
                               dict(self.provenance, lock=lock))
        rows = [self.research_record(bundle, identity) for identity in cases]
        self.rewrite_records(bundle, rows)
        self.assertEqual(self.e.verify_bundle(bundle.path, require_complete=False)["case_count"], 100)
        manifest_path = bundle.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["provenance"]["lock"][0]
        retained_lock = bundle.path / entry["path"]
        for key, invalid in (("rates", {"diagnostic_eta": 0.3, "extra_eta": 0.03}),
                             ("seeds", [2001]), ("arms", ["mir"]),
                             ("architecture", [196, 64, 10]),
                             ("data_manifest_sha256", "0" * 64),
                             ("complete", False), ("no_prior_test_selection", False)):
            with self.subTest(key=key):
                retained_lock.write_text(json.dumps(dict(lock_value, **{key: invalid})))
                entry["sha256"] = self.e.sha256_file(retained_lock)
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises(self.e.VerificationError):
                    self.e.verify_bundle(bundle.path, require_complete=False)
        bundle.close()


    def test_research_pairing_hashes_are_required_and_equal_within_seed(self):
        bundle, rows = self.research_bundle()
        self.assertEqual(self.e.verify_bundle(bundle.path, require_complete=False)["case_count"], 30)
        for key in ("candidate_trace_sha256", "reservoir_trace_sha256", "quota_trace_sha256", "stream_sha256"):
            for mode in ("missing", "mismatch"):
                with self.subTest(key=key, mode=mode):
                    changed = json.loads(json.dumps(rows))
                    if mode == "missing":
                        changed[0].pop(key)
                    else:
                        changed[0][key] = "0" * 64
                    self.rewrite_records(bundle, changed)
                    with self.assertRaises(self.e.VerificationError):
                        self.e.verify_bundle(bundle.path, require_complete=False)
        bundle.close()

    def test_research_ledger_required_and_exact_counters(self):
        bundle, rows = self.research_bundle()
        changed = json.loads(json.dumps(rows))
        changed[0].pop("ledger")
        self.rewrite_records(bundle, changed)
        with self.assertRaises(self.e.VerificationError):
            self.e.verify_bundle(bundle.path, require_complete=False)
        for key in ("committed_base_updates", "fresh_presentations", "current_repeat_presentations",
                    "historical_presentations", "unique_observations", "trainable_scalars",
                    "trainable_raw_bytes", "reservoir_raw_bytes", "virtual_gradients"):
            with self.subTest(key=key):
                changed = json.loads(json.dumps(rows))
                changed[0]["ledger"][key] += 1
                self.rewrite_records(bundle, changed)
                with self.assertRaises(self.e.VerificationError):
                    self.e.verify_bundle(bundle.path, require_complete=False)
        bundle.close()

    def test_research_compute_cap_residual_and_finiteness(self):
        bundle, rows = self.research_bundle()
        for key, value in (("operations_used", 1000001), ("operations_allowed", -1),
                           ("residual_operations", 1), ("operations_used", float("inf")),
                           ("operations_used", True), ("extra_updates", 1)):
            with self.subTest(key=key, value=value):
                changed = json.loads(json.dumps(rows))
                changed[0]["ledger"][key] = value
                self.rewrite_records(bundle, changed)
                with self.assertRaises(self.e.VerificationError):
                    self.e.verify_bundle(bundle.path, require_complete=False)
        bundle.close()

    def test_research_audit_schedule_exactly_eighty_opportunities(self):
        bundle, rows = self.research_bundle()
        audit = bundle.path / rows[0]["case_id"] / "mechanism-audit.jsonl"
        original = audit.read_text()
        for mode in ("missing", "duplicate", "wrong_update"):
            with self.subTest(mode=mode):
                lines = original.splitlines()
                if mode == "missing":
                    lines.pop()
                elif mode == "duplicate":
                    lines[-1] = lines[0]
                else:
                    item = json.loads(lines[0]); item["update"] = 500
                    lines[0] = json.dumps(item)
                audit.write_text("\n".join(lines) + "\n")
                with self.assertRaises(self.e.VerificationError):
                    self.e.verify_bundle(bundle.path, require_complete=False)
        bundle.close()

    def test_research_checkpoint_arrays_shape_dtype_finiteness_and_members(self):
        import numpy as np
        bundle, rows = self.research_bundle()
        path = bundle.path / rows[0]["checkpoints"][0]["path"]
        for mode in ("bytes", "shape", "dtype", "nonfinite", "extra", "missing"):
            with self.subTest(mode=mode):
                arrays = {"p0": np.zeros((196, 32), np.float32), "p1": np.zeros(32, np.float32),
                          "p2": np.zeros((32, 10), np.float32), "p3": np.zeros(10, np.float32)}
                if mode == "bytes":
                    path.write_bytes(b"checksummed but not a model")
                else:
                    if mode == "shape": arrays["p0"] = np.zeros((196, 31), np.float32)
                    if mode == "dtype": arrays["p1"] = np.zeros(32, np.float64)
                    if mode == "nonfinite": arrays["p2"][0, 0] = np.nan
                    if mode == "extra": arrays["unexpected"] = np.zeros(1, np.float32)
                    if mode == "missing": arrays.pop("p3")
                    np.savez(path, **arrays)
                rows[0]["checkpoints"][0]["sha256"] = self.e.sha256_file(path)
                self.rewrite_records(bundle, rows)
                with self.assertRaises(self.e.VerificationError):
                    self.e.verify_bundle(bundle.path, require_complete=False)
        bundle.close()


    def test_research_stream_hash_and_unique_source_ids(self):
        import numpy as np
        bundle, rows = self.research_bundle()
        path = bundle.path / rows[0]["case_id"] / "stream-indices.npy"
        for mode in ("tamper_hash", "duplicate", "short", "float"):
            with self.subTest(mode=mode):
                ids = np.arange(10000, dtype=np.uint64)
                if mode == "duplicate": ids[-1] = ids[0]
                if mode == "short": ids = ids[:-1]
                if mode == "float": ids = ids.astype(np.float64)
                np.save(path, ids, allow_pickle=False)
                rows[0]["stream_sha256"] = "0" * 64 if mode == "tamper_hash" else self.e.sha256_file(path)
                self.rewrite_records(bundle, rows)
                with self.assertRaises(self.e.VerificationError):
                    self.e.verify_bundle(bundle.path, require_complete=False)
        bundle.close()


    def test_research_budget_cannot_be_fabricated_while_internally_consistent(self):
        bundle, rows = self.research_bundle()
        changed = json.loads(json.dumps(rows))
        for row in changed:
            row["ledger"]["operations_allowed"] += 1000
            row["ledger"]["operations_used"] += 1000
        self.rewrite_records(bundle, changed)
        with self.assertRaises(self.e.VerificationError):
            self.e.verify_bundle(bundle.path, require_complete=False)
        changed = json.loads(json.dumps(rows))
        next(row for row in changed if row["arm"] == "uniform-extra-replay")["ledger"]["extra_updates"] += 1
        self.rewrite_records(bundle, changed)
        with self.assertRaises(self.e.VerificationError):
            self.e.verify_bundle(bundle.path, require_complete=False)
        bundle.close()

    def test_research_audit_requires_actual_finite_numeric_results(self):
        bundle, rows = self.research_bundle()
        audit = bundle.path / rows[0]["case_id"] / "mechanism-audit.jsonl"
        original = audit.read_text()
        required = ("selected_mean_harm", "eligible_mean_harm", "mir_highloss_overlap",
                    "selected_minus_shuffled_old_loss", "audit_operations", "candidate_rank_vs_exhaustive_harm")
        for key in required:
            with self.subTest(missing=key):
                entries = [json.loads(line) for line in original.splitlines()]
                entries[0].pop(key)
                audit.write_text("".join(json.dumps(row) + "\n" for row in entries))
                with self.assertRaises(self.e.VerificationError):
                    self.e.verify_bundle(bundle.path, require_complete=False)
        for key, invalid in (("selected_mean_harm", "not numeric"),
                             ("mir_highloss_overlap", 1.1),
                             ("candidate_rank_vs_exhaustive_harm", 2.),
                             ("audit_operations", -1), ("audit_operations", True)):
            with self.subTest(key=key, invalid=invalid):
                entries = [json.loads(line) for line in original.splitlines()]
                entries[0][key] = invalid
                audit.write_text("".join(json.dumps(row) + "\n" for row in entries))
                with self.assertRaises(self.e.VerificationError):
                    self.e.verify_bundle(bundle.path, require_complete=False)
        entries = [json.loads(line) for line in original.splitlines()]
        entries[0]["candidate_rank_vs_exhaustive_harm"] = None
        audit.write_text("".join(json.dumps(row) + "\n" for row in entries))
        self.assertEqual(self.e.verify_bundle(bundle.path, require_complete=False)["case_count"], 30)
        bundle.close()


if __name__ == "__main__":
    unittest.main()
