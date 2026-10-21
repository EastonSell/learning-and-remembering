# Checkpoint availability

The local research run generated 30 original model checkpoints and six seed-11 repeat checkpoints. These are tiny numeric-only NumPy `.npz` archives, licensed MIT. They contain trained parameter arrays, fixed input-group indices, and the validation-selected intervention mapping. Load them with `numpy.load(path, allow_pickle=False)`.

The recommended **text-only publication bundle omits the `.npz` files**. Published metrics and manifests remain complete, including the original SHA-256 of every checkpoint. An absent checkpoint in that bundle is intentional, not a failed training run. The publication manifest enumerates all omissions.

Regenerate all five-seed checkpoints without a model or data download:

```sh
bash scripts/run_pilot.sh --output results/local-reproduction
PYTHONPATH=src python scripts/verify_run.py results/pilot \
  --allow-missing-checkpoints --compare results/local-reproduction
```

This audits available source/text files, compares every numerical metric excluding runtime, and checks regenerated checkpoint hashes against the original manifest when the original binary is omitted. Exact reproducibility is verified on the recorded Python/NumPy environment. Different numerical libraries/platforms may introduce small floating-point differences and should be reported as a new run rather than silently accepted.

To audit the text-only bundle without retraining:

```sh
PYTHONPATH=src python scripts/verify_run.py results/pilot --allow-missing-checkpoints
```

Without that flag, missing binary files correctly cause the audit to fail. When binaries are present, the flag does not skip their verification.
