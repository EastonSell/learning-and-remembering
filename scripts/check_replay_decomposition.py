"""Independently check saved decomposition scalars, without third-party imports.

This narrow check reads only the 16 retained state JSON files and summary.json.
It does not verify release integrity, resample uncertainty, import model code,
load checkpoints/images, or establish any new scientific result. Use the frozen
study's verify_saved.py for its complete integrity/arithmetic/bootstrap checks.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean

SEEDS = (2005, 2006, 2007, 2008)
BLOCKS = (0, 1, 2, 3)
CELLS = ("00", "01", "10", "11")
PANELS = ("old", "new")
COMPARISONS = (
    "mir_minus_highloss", "mir_minus_uniform",
    "random_tie_mir_minus_highloss", "random_tie_contrast_shift",
)
CONTROLS = ("original", "fresh_deployed", "fresh_unit")
ABS_TOL = 1e-12


def factorial(cells):
    """Compute the specified 2x2 simple effects, interaction and reconciliation."""
    if set(cells) != set(CELLS) or not all(math.isfinite(v) for v in cells.values()):
        raise ValueError("four finite factorial cells required")
    d00, d01, d10, d11 = (cells[c] for c in CELLS)
    weight = d10 - d00
    location = d01 - d00
    interaction = d11 - d10 - d01 + d00
    total = d11 - d00
    return {
        "weight_at_original": weight,
        "location_at_deployed": location,
        "interaction": interaction,
        "total_sequential_minus_joint": total,
        "weight_at_updated": d11 - d01,
        "location_at_unit": d11 - d10,
        "reconciliation_residual": total - (weight + location + interaction),
    }


def sign_counts(values):
    """Strict signs: zero, including negative zero, is not a directional effect."""
    values = list(values)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("finite scalar values required")
    return {
        "negative": sum(v < 0 for v in values),
        "zero": sum(v == 0 for v in values),
        "positive": sum(v > 0 for v in values),
    }


def cluster_summary(records):
    """Average four dependent blocks per seed, then four equally weighted seeds."""
    records = list(records)
    identities = [(seed, block) for seed, block, _ in records]
    expected = {(seed, block) for seed in SEEDS for block in BLOCKS}
    if len(identities) != 16 or set(identities) != expected:
        raise ValueError("exactly one row per historical seed/block required")
    state_signs = sign_counts(value for _, _, value in records)
    means = {
        str(seed): fmean(value for s, _, value in records if s == seed)
        for seed in SEEDS
    }
    return {
        "mean": fmean(means.values()),
        "seed_means": means,
        "state_sign_counts": state_signs,
        "seed_sign_counts": sign_counts(means.values()),
        "cluster_count": 4,
        "state_count": 16,
    }


def load_evidence(study):
    """Read only scalar JSON; preserve the closed study directory unchanged."""
    evidence = Path(study) / "evidence"
    rows = []
    for seed in SEEDS:
        for block in BLOCKS:
            path = evidence / f"seed-{seed}-block-{block}.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            if (row["seed"], row["block"]) != (seed, block):
                raise ValueError(f"seed/block identity mismatch in {path.name}")
            rows.append(row)
    saved = json.loads((evidence / "summary.json").read_text(encoding="utf-8"))
    if saved["state_identities"] != [[s, b] for s in SEEDS for b in BLOCKS]:
        raise ValueError("saved summary seed/block inventory mismatch")
    return rows, saved["summaries"]


def recompute(rows):
    """Use raw branch losses, not cached contrasts/components/metrics, as inputs."""
    values = defaultdict(list)
    for row in rows:
        def record(key, value):
            values[key].append((row["seed"], row["block"], value))

        for panel in PANELS:
            contrasts = {comparison: {} for comparison in COMPARISONS}
            for cell in CELLS:
                branches = row["cells"][cell]["branches"]
                loss = lambda name: branches[name][panel + "_loss"]
                mir_highloss = loss("mir") - loss("highloss")
                tie = loss("mir_random_ties") - loss("highloss_random_ties")
                contrasts["mir_minus_highloss"][cell] = mir_highloss
                contrasts["mir_minus_uniform"][cell] = loss("mir") - fmean(
                    loss(f"uniform_{i}") for i in range(4)
                )
                contrasts["random_tie_mir_minus_highloss"][cell] = tie
                contrasts["random_tie_contrast_shift"][cell] = tie - mir_highloss
                for control in CONTROLS:
                    record(f"cell{cell}.mir.{panel}_change_from_{control}",
                           loss("mir") - row["baselines"][control][panel])
            for comparison, cells in contrasts.items():
                prefix = f"{panel}.{comparison}."
                for cell, value in cells.items():
                    record(prefix + "cell" + cell, value)
                effects = factorial(cells)
                if abs(effects.pop("reconciliation_residual")) > 1e-10:
                    raise ValueError("factorial reconciliation failed")
                # Tiny floating residuals are checked against tolerance, not
                # interpreted as substantive signs or resummarized here.
                for component, value in effects.items():
                    record(prefix + component, value)
    if len(values) != 104:
        raise ValueError("expected 104 interpretation-facing scalar outcomes")
    return {key: cluster_summary(records) for key, records in sorted(values.items())}


def compare_saved(actual, saved):
    """Compare scalar means to 1e-12 absolute tolerance and signs/counts exactly."""
    if len(actual) != 104:
        raise ValueError("expected 104 interpretation-facing scalar outcomes")
    for key, summary in actual.items():
        recorded = saved[key]
        for field in ("mean", "seed_means"):
            pairs = [(summary[field], recorded[field])] if field == "mean" else [
                (summary[field][str(s)], recorded[field][str(s)]) for s in SEEDS
            ]
            for computed, expected in pairs:
                if not math.isfinite(expected) or not math.isclose(
                    computed, expected, rel_tol=0.0, abs_tol=ABS_TOL
                ):
                    raise ValueError(f"{key}: {field} mismatch")
        for field in ("state_sign_counts", "seed_sign_counts", "cluster_count", "state_count"):
            if summary[field] != recorded[field]:
                raise ValueError(f"{key}: {field} mismatch")
    return {
        "status": "PASS_narrow_saved_scalar_check",
        "checked_summaries": len(actual),
        "clusters": 4,
        "states": 16,
        "bootstrap_recomputed": False,
        "model_code_imported": False,
        "release_integrity_verified": False,
        "limitations": "Saved scalar arithmetic only, not full release verification or scientific replication.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1]
                        / "studies/replay-update-decomposition")
    args = parser.parse_args()
    try:
        rows, saved = load_evidence(args.study)
        result = compare_saved(recompute(rows), saved)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
