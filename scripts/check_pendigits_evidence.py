"""Fail-closed, read-only checks of the published PenDigits scalar evidence.

This additive checker does not import the frozen experiment, train, download,
run checkpoint inference, or rewrite receipts. PASS means the named checks agree,
not that MIR met its support criterion or that execution history was attested.
All validation remains active with python -O / PYTHONOPTIMIZE.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np


SEEDS = tuple(range(31001, 31011))
ARMS = ("uniform", "high_loss", "mir")
PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9),
         (8, 9), (6, 7), (4, 5), (2, 3), (0, 1))
COUNTS = {
    "actual_steps": 760,
    "virtual_steps": 760,
    "candidate_pre_post_scored": 192000,
    "fresh_presentations": 23980,
    "replay_presentations": 24192,
}
EVIDENCE_FILES = {
    "REPORT.md", "PROTOCOL.md", "experiment.py", "verify_results.py",
    "independent_outcome_audit.py", "freeze.json", "independent-review.json",
    "execution.log", "results/summary.json", "results/metrics.json",
    "results/verification.json", "results/independent-audit.json",
    "results/COMPLETE.json", "results/seed-results.csv",
}
ABS_TOL = 1e-12
DEFAULT_STUDY = Path(__file__).resolve().parents[1] / "studies/pendigits-recurrent-replay"


def require(condition, message):
    """Do not replace with assert: Python optimization must not disable checks."""
    if not condition:
        raise ValueError(message)


def number(value, label, low=None, high=None):
    require(type(value) in (int, float) and math.isfinite(value),
            f"{label}: expected finite number, not boolean or string")
    require(low is None or value >= low, f"{label}: below {low}")
    require(high is None or value <= high, f"{label}: above {high}")
    return value


def close(saved, actual, label):
    number(saved, label)
    require(math.isclose(saved, float(actual), rel_tol=0.0, abs_tol=ABS_TOL),
            f"{label}: saved value disagrees with recomputation")


def read_bytes(study, name):
    # Names are fixed by this checker, never taken as arbitrary catalog paths.
    path = study / name
    require(not any(p.is_symlink() for p in (path, *path.parents)
                    if p != study.parent and study.parent in p.parents),
            f"{name}: symlink evidence is not accepted")
    require(path.is_file(), f"{name}: missing regular evidence file")
    return path.read_bytes()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite(token):
    raise ValueError(f"non-finite JSON number: {token}")


def finite_json_float(token):
    value = float(token)
    if not math.isfinite(value):
        reject_nonfinite(token)
    return value


def load_json(study, name):
    return json.loads(read_bytes(study, name), object_pairs_hook=unique_object,
                      parse_constant=reject_nonfinite, parse_float=finite_json_float)


def check_integrity(study):
    catalog = load_json(study, "compact-evidence-manifest.json")
    require(isinstance(catalog, dict) and set(catalog) == EVIDENCE_FILES,
            "catalog inventory must contain exactly the 14 published evidence paths")
    for name, expected in sorted(catalog.items()):
        require(isinstance(expected, str) and len(expected) == 64
                and set(expected) <= set("0123456789abcdef"),
                f"{name}: invalid SHA-256 in catalog")
        require(hashlib.sha256(read_bytes(study, name)).hexdigest() == expected,
                f"{name}: SHA-256 mismatch")
    freeze = load_json(study, "freeze.json")
    review = load_json(study, "independent-review.json")
    for key, name in (("source_sha256", "experiment.py"),
                      ("protocol_sha256", "PROTOCOL.md")):
        require(catalog[name] == freeze[key] == review[key],
                f"{name}: freeze/review hash mismatch")
    require(catalog["independent-review.json"] == freeze["review_sha256"],
            "independent-review.json: freeze hash mismatch")
    marker = study / "results/INCOMPLETE"
    require(not marker.exists() and not marker.is_symlink(),
            "INCOMPLETE marker present: this is not a completed run")
    complete = load_json(study, "results/COMPLETE.json")
    for key, expected in (("cases", 30), ("checkpoints", 300)):
        require(type(complete[key]) is int and complete[key] == expected,
                f"completion {key}: expected {expected}")
    require(complete["source_sha256"] == freeze["source_sha256"],
            "completion source does not match freeze")
    return len(catalog)


def check_rows(rows):
    require(isinstance(rows, list) and len(rows) == 30, "expected exactly 30 cases")
    by = {}
    for row in rows:
        seed, arm = row["seed"], row["arm"]
        require(type(seed) is int and seed in SEEDS and isinstance(arm, str)
                and arm in ARMS, "invalid case identity")
        identity = (seed, arm)
        require(identity not in by, f"duplicate case identity: {identity}")
        by[identity] = row
        label = f"seed {seed} {arm}"
        boundaries = row["boundaries"]
        require(isinstance(boundaries, list) and len(boundaries) == 10,
                f"{label}: expected 10 boundaries")
        seen = set()
        for phase, (boundary, pair) in enumerate(zip(boundaries, PAIRS)):
            where = f"{label} phase {phase}"
            require(type(boundary["phase"]) is int and boundary["phase"] == phase,
                    f"{where}: phase order mismatch")
            current = boundary["current_pair"]
            require(isinstance(current, list) and all(type(c) is int for c in current)
                    and current == list(pair), f"{where}: current_pair schedule mismatch")
            recalls = boundary["recalls"]
            require(isinstance(recalls, list) and len(recalls) == 10,
                    f"{where}: expected 10 recalls")
            for value in recalls:
                number(value, where + " recalls", 0, 1)
            close(boundary["macro"], np.mean(recalls), where + " macro")
            close(boundary["acquisition"], np.mean([recalls[c] for c in pair]),
                  where + " acquisition")
            seen.update(pair)
            old = sorted(seen - set(pair))
            if old:
                close(boundary["old_accuracy"], np.mean([recalls[c] for c in old]),
                      where + " old_accuracy")
            else:
                require(boundary["old_accuracy"] is None,
                        f"{where}: old_accuracy must be null before any old class")
            # Original labels/probabilities are not bundled. Only ranges, not
            # micro-accuracy or CE inference correctness, can be checked here.
            number(boundary["micro"], where + " micro", 0, 1)
            number(boundary["ce"], where + " ce", 0)
        close(row["final_macro"], boundaries[-1]["macro"], label + " final_macro")
        close(row["mean_acquisition"], np.mean([b["acquisition"] for b in boundaries]),
              label + " mean_acquisition")
        counts = row["counts_per_arm"]
        require(isinstance(counts, dict) and set(counts) == set(COUNTS)
                and all(type(counts[key]) is int and counts[key] == expected
                        for key, expected in COUNTS.items()),
                f"{label}: counts_per_arm differs from frozen run accounting")
    require(set(by) == {(seed, arm) for seed in SEEDS for arm in ARMS},
            "case identity coverage mismatch")
    return by


def check_csv(study, by):
    reader = csv.DictReader(io.StringIO(read_bytes(study, "results/seed-results.csv")
                                       .decode("utf-8"), newline=""))
    require(reader.fieldnames == ["seed", "arm", "final_macro_accuracy",
                                  "mean_current_pair_acquisition"],
            "CSV column inventory mismatch")
    seen = set()
    for row in reader:
        require(None not in row and all(value is not None for value in row.values()),
                "CSV row has missing or extra fields")
        identity = (int(row["seed"]), row["arm"])
        require(identity in by and identity not in seen, "CSV case identity mismatch")
        seen.add(identity)
        close(float(row["final_macro_accuracy"]), by[identity]["final_macro"],
              "CSV final_macro_accuracy")
        close(float(row["mean_current_pair_acquisition"]), by[identity]["mean_acquisition"],
              "CSV mean_current_pair_acquisition")
    require(seen == set(by), "CSV case identity coverage mismatch")
    return len(seen)


def check_summary(summary, by):
    differences = np.array([by[seed, "mir"]["final_macro"]
                            - by[seed, "uniform"]["final_macro"] for seed in SEEDS])
    acquisitions = np.array([by[seed, "mir"]["mean_acquisition"]
                             - by[seed, "uniform"]["mean_acquisition"] for seed in SEEDS])
    ci = np.quantile(np.random.default_rng(20261025).choice(
        differences, size=(10000, len(SEEDS)), replace=True).mean(1), [.025, .975])
    for key, actual in (("mir_minus_uniform", differences),
                        ("paired_seed_bootstrap_95", ci)):
        saved = summary[key]
        require(isinstance(saved, list) and len(saved) == len(actual),
                f"{key}: wrong length")
        for index, (value, expected) in enumerate(zip(saved, actual)):
            close(value, expected, f"{key}[{index}]")
    close(summary["mean_difference"], differences.mean(), "mean_difference")
    close(summary["mean_acquisition_difference"], acquisitions.mean(),
          "mean_acquisition_difference")
    means = summary["mean_final_macro"]
    require(isinstance(means, dict) and set(means) == set(ARMS),
            "mean_final_macro arm inventory mismatch")
    for arm in ARMS:
        close(means[arm], np.mean([by[seed, arm]["final_macro"] for seed in SEEDS]),
              f"mean_final_macro {arm}")
    components = {
        "mean_gain_at_least_one_point": bool(differences.mean() >= .01),
        "interval_strictly_above_zero": bool(ci[0] > 0),
        "acquisition_drop_at_most_two_points": bool(acquisitions.mean() >= -.02),
    }
    support = all(components.values())
    require(type(summary["support"]) is bool and summary["support"] == support,
            "support disagrees with the three frozen gates")
    require(type(summary["test_rows"]) is int and summary["test_rows"] == 3498,
            "test_rows must be the recorded 3498")
    number(summary["wall_seconds"], "reported wall_seconds", 0, 600)
    require(type(summary["peak_rss_bytes"]) is int,
            "reported peak_rss_bytes must be an integer")
    number(summary["peak_rss_bytes"], "reported peak_rss_bytes", 1, 1024 ** 3)
    return {
        "mir_minus_uniform_pp": float(100 * differences.mean()),
        "ci95_pp": (100 * ci).tolist(),
        "acquisition_difference_pp": float(100 * acquisitions.mean()),
        "support_components": components,
        "support": support,
    }


def verify(study):
    study = Path(study).absolute()
    checked_files = check_integrity(study)
    by = check_rows(load_json(study, "results/metrics.json"))
    csv_rows = check_csv(study, by)
    outcome = check_summary(load_json(study, "results/summary.json"), by)
    return {
        "status": "PASS", "cases": len(by), "boundaries": 10 * len(by),
        "hashed_files": checked_files, "csv_rows": csv_rows,
        "bootstrap_recomputed": True, "model_code_imported": False,
        "numpy_version": np.__version__,
        "scope": "published hashes, completion consistency, saved scalar arithmetic, "
                 "CSV agreement and paired-seed bootstrap only; no inference or training",
        "limitations": "Consistency only, not exact-release authentication, execution "
                       "attestation, original-checkpoint verification, "
                       "or independent validation of producer-measured resources. "
                       "An interval crossing zero establishes neither equivalence nor inferiority; "
                       "high-loss remains descriptive.",
        **outcome,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY,
                        help="read-only published study directory")
    args = parser.parse_args(argv)
    try:
        result = verify(args.study)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, csv.Error) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, allow_nan=False))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
