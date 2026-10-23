# Historical replay update decomposition: weighting × gradient location

This saved-evidence study separates two parts of a one-step replay update on sixteen fixed states from four reused historical Fashion-MNIST seed/stream clusters. It is an additive follow-up to the [original JOINT diagnostic](../replay-joint-update-diagnostic/README.md), not its replacement or an independent confirmation. The [portable reproduction](../replay-joint-update-reproduction/README.md) reproduces that earlier diagnostic, not this new factorial cohort.

## Read the adverse result first

Deployed JOINT MIR increases absolute old-class loss by **+0.905519 nats/example**, descriptive 95% interval [+0.526345, +1.480879], and is worse than the original checkpoint in **all 16 states**. It therefore does not prevent immediate old-class damage.

Its old-class MIR-minus-high-loss contrast averages −0.025069, interval [−0.051397, −0.004300]. Negative favors MIR, but the interval's upper bound does **not** clear the frozen −0.005 engineering margin. JOINT loses to high-loss in 7/16 states. Only three of four seed means show the earlier negative-to-positive endpoint reversal; seed 2006 has positive contrasts at both endpoints.

There is a fresh-learning tradeoff: JOINT MIR raises new-class loss by +0.335974 versus unit-weight fresh-only SGD, interval [+0.162430, +0.488773], with a cost in 12/16 states. Against the weaker, deployed-weight fresh-only control, its mean new loss instead decreases by −0.057774. Against high-loss, the new-class contrast interval crosses zero and MIR is worse in 11/16 states. These are different controls, not interchangeable plasticity claims.

## What the 2×2 adds

For old-class MIR-minus-high-loss loss, the four cell means are:

| | Replay gradient at original state | Replay gradient after weight-matched fresh step |
|---|---:|---:|
| Deployed normalized weights | D00 = −0.025069 | D01 = −0.070031 |
| Unit fresh and replay weights | D10 = −0.131991 | D11 = +0.340123 |

At the aggregate reference cell, either factor alone leaves the MIR contrast negative. The combined endpoint becomes positive through a large positive interaction:

- Weighting effect at original replay location, W = −0.106923.
- Location effect at deployed weights, L = −0.044963.
- Interaction, I = +0.517077, interval [+0.164894, +1.034945].
- Endpoint difference, D11 − D00 = +0.365192 = W + L + I.

This is a local algebraic decomposition of specified interventions. The weighting factor combines relative ratio and total scale; it does not isolate either. The replay displacement is weight-matched, not identical at both weighting levels. The interaction is positive in all four seed means but negative in 3/16 states. There is no universal individual-state mechanism, robust efficacy conclusion, brain mapping or biological validation.

Read the [plain-language explanation](docs/PLAIN_LANGUAGE.md), [methods and locked estimands](docs/METHODS.md), and [complete results](docs/RESULTS.md). The four exact CSV exports retain both panels, all four comparison families, all signed states and seed means, and every one of the 350 scalar summaries:

- [All 350 summaries](analysis/all-350-summary-metrics.csv).
- [All component summaries](analysis/aggregate-components.csv).
- [All signed state components](analysis/signed-state-components.csv).
- [All signed seed components](analysis/signed-seed-components.csv).
- [Full saved analysis, labeled public derivative](analysis/saved-analysis.public.json).

Random-exact-tie comparisons are retained, not selected away: no MIR or high-loss selection changes in these states, so their differences and interactions are exactly zero. Four uniform replay draws and tie branches do not create extra independent samples.

## Verify the saved evidence, not a new experiment

Recorded verification environment: Python 3.12.14 and NumPy 2.3.5, Linux. NumPy is already required for the fixed scalar bootstrap; the scripts install and download nothing. From this study directory:

    env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B verify_saved.py
    env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest discover -s tests -v

Invoke verify_saved.py by its full path to verify from another working directory. The verifier writes only stdout; keep redirected output outside this closed package. The tests copy saved fixtures into temporary directories and exercise inventory, scalar, interval, nonfinite/duplicate JSON and symlink failures. They are software tests, not scientific samples. No reference source is imported and no model loss, gradient, image, checkpoint array, training, continuation, parameter comparison or scientific numerical run is regenerated. The bootstrap only recounts already-saved scalars.

The verifier checks the closed public allowlist, original retained artifact hashes, recorded checkpoint/state identities, panel/selection memberships, signed score ranking, controls, all cell/component arithmetic, all 350 summaries and their fixed four-cluster intervals, and all CSV exports. It checks the recorded numerical-equivalence assertions and loss-difference arithmetic, not parameter arrays. [Original independent-review findings](analysis/independent-review.public.json) are a labeled selection from the saved review, not independent live execution attestation.

## Exact bytes, derivatives and missing inputs

Sixteen raw state files, summary.json and COMPLETE.json are exact original bytes. So are the four CSVs, reference mechanics/kernel/license and three small saved process-output files. [transformations.json](transformations.json) records original/public hashes, exact copies, deliberately labeled derivatives and new documentation/software. [PUBLIC-FILES.json](PUBLIC-FILES.json) is the closed release allowlist, excluding only itself. Hashes prove internal integrity relative to that manifest, not independent authenticity of omitted originals.

The unchanged original COMPLETE describes the **original 19-artifact directory**, totaling **3,175,082 bytes**, including the original run-input freeze. It is not the byte count of this public package. The original freeze is undistributed; its SHA256 commitment is 8c70ca1b3a7ea529bddadd360ee78a40be5115f1dc5650aed2305bee2e85876c. The separately named run-input-freeze.public.json replaces absolute inventory/checkpoint paths with opaque identity-only aliases, keeps all 65 hash bindings, and preserves all other original metadata. Its bytes and hash differ. The verifier checks the 17 unchanged receipt-listed result files, the exact original completion marker, and the explicit omitted-freeze-to-public-derivative commitment. It does not pretend the public derivative is the executed freeze.

The source-freeze's false execution/image-loading flags describe its pre-execution creation state, not the later experiment's absence. The saved original evidence records one scientific invocation, observed external exit 0, empty stderr, 672 gradient calls, 1,600 candidate-example forwards and 2,408,000 panel-example forwards. Resource samples were taken before completion writing, not at process termination: CPU 3.344052105 s, wall 3.365502604 s, RSS high-water 96,952,320 bytes. This release supplies saved evidence, not an independent process witness or syscall trace.

**Full scientific execution reproduction is not packaged or verified here.** Exact cohort-2005–2008 checkpoint/image/stream/split/duplicate metadata, the entire source/launch/input chain and original address-bearing provenance are not supplied. The earlier portable package concerns seeds 2001–2004 and must not be silently substituted. Reference mechanics and kernel files support code inspection only; no scientific launcher is included. No public freeze reconstruction, new execution, data acquisition or changed scientific result was performed for this release.

## Interpretation boundary

Seeds 2005–2008 are the next mechanically chosen historical lock positions, with all four eligible transitions per seed and eta 0.3 unchanged. This is historical/post-hoc work reusing data and validation. The uncertainty units are **four clusters**, not sixteen dependent states, repeated replay draws or 350 outcome summaries. The fixed 10,000-draw cluster bootstrap is descriptive, not powered independent confirmation. No pooling with the earlier audit upgrades it to fresh evidence. Future retention, new domains, alternative rates/models and a ratio-versus-scale separation remain untested.

Original code and documentation retain the [MIT notice](LICENSE). The Fashion-MNIST dataset and dependencies retain their upstream rights; raw data is not redistributed in this package. Existing studies and their adverse findings remain unchanged.
