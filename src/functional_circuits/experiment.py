"""Run the complete CPU pilot: python -m functional_circuits.experiment."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
from .data import generate
from .models import Model, KINDS, train
from .metrics import choose_mapping, evaluate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows):
    output = []
    for kind in KINDS:
        for split in ("id", "balanced", "conflict"):
            group = [r for r in rows if r["model"] == kind and r["split"] == split]
            if not group:
                continue
            item = {"model": kind, "split": split, "seeds": len(group)}
            for key in group[0]:
                if key in ("seed", "model", "split", "mapping"):
                    continue
                if isinstance(group[0][key], (float, int)):
                    v = np.array([r[key] for r in group], float)
                    item[key+"_mean"] = float(v.mean())
                    item[key+"_std"] = float(v.std(ddof=1)) if len(v)>1 else 0.0
            output.append(item)
    return output


def paired_comparisons(rows):
    results = []
    # Two-sided t critical values; five seeds use df=4. No scipy dependency.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    for split in ("id", "balanced", "conflict"):
        by = {(r["model"], r["seed"]): r for r in rows if r["split"] == split}
        seeds = sorted({seed for model, seed in by if model == "functional"})
        for other in KINDS:
            if other == "functional":
                continue
            for metric in ("accuracy", "memory_iia", "memory_iia_changed"):
                v = np.array([by[("functional", s)][metric]-by[(other, s)][metric] for s in seeds])
                if len(v) not in critical:
                    half = None
                else:
                    half = float(critical[len(v)]*v.std(ddof=1)/np.sqrt(len(v)))
                results.append({"contrast": "functional_minus_"+other, "split": split, "metric": metric,
                                "paired_differences": v.tolist(), "mean": float(v.mean()),
                                "descriptive_t95_halfwidth": half})
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/pilot")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--test-size", type=int, default=2048)
    p.add_argument("--validation-size", type=int, default=512)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=.01)
    args = p.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        p.error("Seeds must be unique")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"manifest.json").exists():
        p.error("Output already contains a run. Choose a new --output to preserve results.")
    (out/"checkpoints").mkdir(exist_ok=True)
    started = time.perf_counter()
    rows, curves, split_hashes = [], {}, {}
    seed_schedule = {}
    for seed in args.seeds:
        schedule = {"train": seed*100+1, "validation": seed*100+2, "validation_donors": seed*100+3,
                    "id": seed*100+4, "balanced": seed*100+5, "conflict": seed*100+6, "test_donors": seed*100+7}
        seed_schedule[str(seed)] = schedule
        training = generate(args.train_size, schedule["train"], .8)
        validation = generate(args.validation_size, schedule["validation"], .5)
        vd = generate(args.validation_size, schedule["validation_donors"], .5)
        tests = {name: generate(args.test_size, schedule[name], agreement)
                 for name, agreement in (("id", .8), ("balanced", .5), ("conflict", 0.))}
        td = generate(args.test_size, schedule["test_donors"], .5)
        split_hashes[str(seed)] = {name: data.fingerprint() for name, data in
                                 {"train": training, "validation": validation, "validation_donors": vd,
                                  "test_donors": td, **tests}.items()}
        for kind in KINDS:
            t = time.perf_counter()
            model = Model(kind, seed, args.width)
            losses = train(model, training, seed+10000, args.epochs, args.batch_size, args.lr)
            train_seconds = time.perf_counter()-t
            mapping, metadata = choose_mapping(model, validation, vd)
            curves[f"{kind}/{seed}"] = losses
            checkpoint = {**model.params, "group0": model.groups[0], "mapping": np.array(mapping)}
            if len(model.groups)>1:
                checkpoint["group1"] = model.groups[1]
            np.savez_compressed(out/"checkpoints"/f"{kind}_{seed}.npz", **checkpoint)
            for split, data in tests.items():
                row = {"model": kind, "seed": seed, "split": split, "parameters": model.parameter_count,
                       "training_seconds": train_seconds, "final_train_loss": losses[-1]["loss"],
                       "mapping": mapping, **metadata, **evaluate(model, data, td, mapping)}
                rows.append(row)
            print(f"seed={seed} {kind:17s} params={model.parameter_count:3d} "
                  f"balanced_acc={rows[-2]['accuracy']:.4f} balanced_iia={rows[-2]['memory_iia']:.4f}", flush=True)
    summaries, paired = summary(rows), paired_comparisons(rows)
    for name, obj in (("metrics.json", rows), ("summary.json", summaries),
                      ("paired_comparisons.json", paired), ("learning_curves.json", curves)):
        (out/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with (out/"metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    root = Path(__file__).resolve().parents[2]
    code_paths = sorted(path for folder in ("src", "tests", "scripts") for path in (root/folder).rglob("*.py"))
    manifest = {"experiment": "functional-circuits-pilot-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": vars(args), "command": "python -m functional_circuits.experiment " + " ".join(sys.argv[1:] if argv is None else argv), "models": list(KINDS), "python": platform.python_version(),
                "numpy": np.__version__, "platform": platform.platform(),
                "blas_threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                "elapsed_seconds": time.perf_counter()-started, "seed_schedule": seed_schedule,
                "dataset_sha256": split_hashes, "source_sha256": {str(path.relative_to(root)): digest(path) for path in code_paths},
                "artifact_sha256": {str(path.relative_to(out)): digest(path) for path in sorted(out.rglob("*")) if path.is_file()},
                "model_downloads": [], "external_datasets": [], "synthetic_data_license": "MIT",
                "original_code_license": "MIT", "source_access_date": "2026-10-21",
                "tuning": "No hyperparameter search. Validation used only for intervention mapping.",
                "uncertainty": "Paired seed differences and descriptive t intervals; not corrected for multiple comparisons."}
    lines = ["# Measured pilot results", "", f"{len(args.seeds)} seeds; {args.epochs} epochs; all values below are means +/- sample SD.", "",
             "| Model | Params | ID accuracy | Accuracy on 50%-agreement split | Conflict accuracy | Balanced memory IIA | Changed-case IIA |", "|---|---:|---:|---:|---:|---:|---:|"]
    for kind in KINDS:
        d = {r["split"]: r for r in summaries if r["model"] == kind}
        fmt = lambda split, metric: f"{d[split][metric+'_mean']:.3f} ± {d[split][metric+'_std']:.3f}"
        lines.append(f"| {kind} | {int(d['id']['parameters_mean'])} | {fmt('id', 'accuracy')} | {fmt('balanced', 'accuracy')} | {fmt('conflict', 'accuracy')} | {fmt('balanced', 'memory_iia')} | {fmt('balanced', 'memory_iia_changed')} |")
    lines += ["", "IIA is activation interchange intervention accuracy. Changed-case IIA excludes counterfactuals whose correct answer was unchanged.",
              "Interpretation requires docs/results.md: architectural role assignment, different patch granularities, and perfect shared retrieval limit conclusions.", ""]
    (out/"summary.md").write_text("\n".join(lines))
    manifest["artifact_sha256"]["summary.md"] = digest(out/"summary.md")
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Saved {out}; {manifest['elapsed_seconds']:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
