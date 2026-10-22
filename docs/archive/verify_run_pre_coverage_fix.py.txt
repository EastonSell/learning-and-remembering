"""Audit hashes and optionally compare deterministic numerical outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--allow-missing-checkpoints", action="store_true",
                        help="Allow intentionally omitted .npz files, never missing text/source artifacts")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((args.run/"manifest.json").read_text())
    omitted = []
    for key, base in (("source_sha256", root), ("artifact_sha256", args.run)):
        for name, expected in manifest[key].items():
            path = base/name
            if not path.exists() and args.allow_missing_checkpoints and key == "artifact_sha256" and name.startswith("checkpoints/") and name.endswith(".npz"):
                omitted.append(name)
                continue
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            if got != expected:
                raise AssertionError(f"Hash mismatch: {name}")
    if args.compare:
        other = json.loads((args.compare/"metrics.json").read_text())
        current = json.loads((args.run/"metrics.json").read_text())
        lookup = {(r["model"], r["seed"], r["split"]): r for r in current}
        for row in other:
            key = (row["model"], row["seed"], row["split"])
            expected = lookup[key]
            for name, value in row.items():
                if name == "training_seconds":
                    continue
                if expected[name] != value:
                    raise AssertionError(f"Numerical mismatch: {key}, {name}")
        for path in (args.compare/"checkpoints").glob("*.npz"):
            reference = args.run/"checkpoints"/path.name
            if not reference.exists() and args.allow_missing_checkpoints:
                key = "checkpoints/"+path.name
                expected_hash = manifest["artifact_sha256"][key]
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                    raise AssertionError(f"Regenerated checkpoint hash differs: {path.name}")
                continue
            with np.load(path, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
                if a.files != b.files:
                    raise AssertionError("Checkpoint fields differ")
                for name in a.files:
                    np.testing.assert_array_equal(a[name], b[name])
        print(f"Exact deterministic match for {len(other)} metric rows and corresponding checkpoints.")
    if omitted:
        print(f"Intentionally omitted local checkpoints: {len(omitted)}. Regenerate with the recorded command.")
    print(f"Verified {len(manifest['source_sha256'])} source hashes and {len(manifest['artifact_sha256'])-len(omitted)} artifact hashes.")


if __name__ == "__main__":
    main()
