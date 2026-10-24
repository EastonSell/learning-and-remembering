# Portable checks for saved replay-decomposition arithmetic

The [frozen study verifier](../studies/replay-update-decomposition/verify_saved.py)
already checks all 350 summaries, the fixed seed-cluster bootstrap, CSV exports,
and release integrity. Do not replace or modify that verifier or its closed
package to run these supplemental checks.

The top-level checker adds a smaller, independent, Python-standard-library route
for arithmetic and interpretation regression checks. It reads the existing 16
raw state JSON files and `summary.json`, not cached state metrics or reference
model code. It installs nothing and does not train, infer, resample a bootstrap,
acquire data, or produce a new scientific observation.

From the repository root, with Python 3.11 or newer:

```sh
python -B -S scripts/check_replay_decomposition.py
python -B -S -m unittest discover -s tests -p test_replay_decomposition_arithmetic.py -v
```

`-S` disables third-party site packages; NumPy is not required. `-B` avoids
bytecode writes. The commands write only stdout/stderr. For a retained evidence
copy elsewhere, pass `--study /path/to/replay-update-decomposition` to the checker
or set `REPLAY_DECOMPOSITION_STUDY` for the tests. Do not redirect output inside
the frozen study directory.

## What is checked

- All four contrast families on both panels: MIR minus high-loss, MIR minus the
  mean of four uniform draws, random-tie MIR minus random-tie high-loss, and the
  random-tie contrast shift.
- Four cells and six substantive factorial effects per panel/comparison: 80
  summaries. The identity `D11 - D00 = W + L + I` is also checked per state.
- MIR's changes from original, deployed-weight fresh, and unit-weight fresh
  controls at all four cells on both panels: 24 additional summaries.
- Four dependent blocks within each historical seed, then four equally weighted
  seed means. Duplicate, missing, extra, or substituted seed/block rows fail.
- Means and seed means within absolute tolerance `1e-12`; state/seed sign counts
  and sample/cluster counts exactly. Reconciliation residuals are bounded by
  `1e-10`, not interpreted as directional scientific effects.

The 104-summary scope is intentional. This is not another full-release verifier.
The other 246 summaries, manifest/hash checks, selection membership/ranking,
numerical-equivalence receipts, CSV exports and bootstrap reconstruction remain
the responsibility of the frozen verifier. The existing saved interval is read
only to guard its interpretation; its correctness is not independently
re-established here.

## Regression-protected interpretation

For the retained cohort only, negative MIR-minus-comparator loss favors MIR:

- Old-panel JOINT's mean is about −0.025069, but MIR loses in 7/16 states and one
  of four seed means. The mean is not an every-state effect.
- The isolated weighting/location effects are favorable on average but adverse
  in 6/16 and 7/16 states, respectively.
- Interaction is positive in all four seed means but negative in 3/16 states.
- Absolute old-panel damage remains positive in all 16 JOINT MIR states, despite
  its favorable mean contrast against high-loss.
- New-panel mean benefit relative to deployed-weight fresh-only and mean cost
  relative to unit-weight fresh-only refer to different controls. Against
  high-loss, the new-panel mean favors MIR although 11/16 states do not.
- The recorded old-panel JOINT interval excludes zero but does not clear the
  −0.005 engineering margin. A point mean beyond the margin is not an
  interval-level result.

These checks preserve existing results and their limitations. Four reused
historical clusters are not sixteen independent observations, new confirmation,
future-retention evidence, or a universal causal mechanism. See the
[methods](../studies/replay-update-decomposition/docs/METHODS.md) and
[complete results](../studies/replay-update-decomposition/docs/RESULTS.md).
