# Scientific method and locked estimands

This is a new public explanation of the accepted historical/post-hoc factorial diagnostic, not the original protocol or a newly executed freeze. Its source commitments are recorded in transformations.json and the identity-only public freeze derivative.

## State, data and selection

Use exactly historical seeds 2005, 2006, 2007 and 2008, positions 4:8 in the pre-existing lock order after the earlier diagnostic's positions 0:4. Include all uniform-origin block transitions 0–3, sixteen states in four seed/stream clusters. No outcome filtering, state substitution, pooling into confirmation, eta search or favorable retry follows from this analysis. The fixed learning rate is 0.3. These checkpoints are additional to the earlier local diagnostic but reuse historical data, trajectories and validation.

At each retained checkpoint, reconstruct the reservoir by its original admissions through stream position 2000 × (block + 1), without retraining. Use the next sixteen fresh observations before admission, with update offset 500 × (block + 1). The common candidate pool has fifty class-balanced resident observations and each replay selection has exactly ten, under the deployed class quota. The named SeedSequence domain is [7310501, seed, block, domain, draw]: domain 1 for candidates, 2 for exact-tie priority, and 3 for four uniform replay draws. These are new named diagnostic RNG draws, not reconstructed historical deployment draws.

Compute all selectors once at the common original state. High-loss ranks current losses; MIR ranks signed loss changes after the original full-rate fresh virtual step. Keep negative and zero scores. Retain source-ID tie-breaking, randomized-exact-tie MIR/high-loss with a common priority permutation, and four within-class uniform draws. Strict unequal scores are never reordered by tie sensitivity. Record class cutoffs, above/equal counts, selected/rejected tie IDs and near-cutoff values within 1e-7. No selector is recomputed inside a factor branch.

Old and next-class panels use all 500 validation members per relevant class, with equal class weighting for macro cross-entropy. Validation panels are evaluator-only and disjoint from fresh/resident training memberships. Original train/validation raw/transformed duplicate-audit identities remain disjoint. The executed design does not use official TEST inputs. The public saved-only verifier checks packaged memberships and identities, not original image bytes, training histories or runtime I/O.

## Four independent, discarded one-step branches

Let gF and gR be mean gradients over the sixteen fresh and ten selected replay examples, eta = 0.3, a = 16/26 and b = 10/26. All four cells share the same original parameter bytes and selected IDs.

- 00: theta − eta [a gF(theta) + b gR(theta)]. Authoritative output is the actual deployed concatenated fresh-plus-replay update, not a substituted separate-gradient approximation.
- 01: theta − eta a gF(theta) − eta b gR(theta − eta a gF(theta)).
- 10: theta − eta [gF(theta) + gR(theta)].
- 11: theta − eta gF(theta) − eta gR(theta − eta gF(theta)). Authoritative output is checked against the exact earlier fresh-then-replay implementation.

Cell 11 and its direct sequential reference must be bitwise equal. The unit-weight fresh control and direct fresh reference must also be bitwise equal. The mathematical separate-gradient version of 00 is only a numerical-equivalence check, not a fifth scientific cell. The fixed parameter tolerance is abs 2e-6 + rel 2e-6 × abs(reference); panel-loss tolerance is abs 2e-5 + rel 2e-6 × abs(actual00 loss). A violation invalidates the bundle. Reported maximum parameter discrepancy is 2.384185791015625e-7, maximum tolerance fraction 0.05147882711875785, and maximum panel-loss discrepancy 6.177425384557012e-7. Those execution-time checks are preserved assertions; this publication does not deserialize parameters or repeat them.

The weight factor jointly changes fresh/replay ratio and total gradient scale. Replay's updated evaluation point uses the corresponding weight-matched fresh displacement. Thus this is a weighting-by-location decomposition, not clean separation of pure replay ratio, pure scale, step count, all floating-point implementation details or a biological mechanism.

## Contrasts and all controls

For each old/new panel and each comparison family, D_wl is MIR loss minus comparator loss; negative favors MIR. Comparators are source-ID high-loss, the mean of four uniform replay draws, randomized-tie high-loss (against randomized-tie MIR), and the randomized-tie contrast shift relative to source-ID contrast. Do not treat tie/uniform branches as independent scientific samples.

For each state and comparison:

- W = D10 − D00: weighting effect at original replay location.
- L = D01 − D00: location effect at deployed weights.
- I = D11 − D10 − D01 + D00: interaction.
- T = D11 − D00 = W + L + I, reconciled within abs 1e-10 nats.
- W_at_updated = D11 − D01 and L_at_unit = D11 − D10: the other conditional simple effects.

Signed cells, not W/L labels alone, decide whether a selector contrast changes sign. Retain every branch's absolute old/new loss and change from original theta and both fresh-only controls: the deployed-weight step eta × a, and full unit-weight step eta. Also retain start/end hashes, replay-gradient location hashes, selected source IDs and numerical checks. A favorable selector-relative contrast does not erase positive original-relative old loss.

## Aggregation and uncertainty

Average the four blocks within each seed, then equally average the four seed means. The fixed descriptive bootstrap uses 10,000 resamples of the four seed means, NumPy PCG64(7310301), with percentile endpoints 2.5 and 97.5. Never resample individual states, examples, factor cells or replay draws as independent observations. Retain state and seed sign counts, all seed means, all signed state rows, and all 350 fixed scalar summaries. The frozen engineering margin is 0.005 nats, with 0.001/0.010 sensitivities. These are descriptive yardsticks, not confirmatory success thresholds.

JOINT's interval lies below zero but its upper endpoint −0.0043000206432646584 does not clear −0.005 or −0.010. It clears only the looser −0.001 yardstick. Its point mean exceeding 0.005 in magnitude is not an interval-level margin result.

## Bounded original execution and evidence status

The saved records describe one scientific process, no persistent training updates and no scientific retry. There are sixteen states × (two fresh gradients + eight selected sets × five gradients) = 672 gradient calls, 1,600 candidate-example forwards and 2,408,000 evaluator-example forwards. The original terminal bundle contains nineteen artifacts including COMPLETE, totaling 3,175,082 bytes. Saved external exit is zero and stderr is empty. Independent saved-result review recomputed all 350 scalar summaries with maximum discrepancy 0.0. Review accepted the provenance envelope and arithmetic, not an independent model rerun or unseen efficacy result.

The resource contract used one thread, CPU soft/hard backstops 180/181 seconds, 300-second signal timer, 768 MiB address space and observed RSS gate, 32 MiB per-file size and bounded aggregate artifact admission with terminal reserves, plus an external TERM330s/KILL+5s backstop. Sampled RSS/output gates and signal timers are not hard aggregate-RSS or exact-wall guarantees. The preserved CPU/wall/RSS samples precede completion write; they are not final process measurements.

The original scientific launcher, cohort-specific execution inputs and entire private provenance chain are not packaged here. The reference mechanics/kernel are exact code for inspection only. Public verification reads saved JSON/CSV and redoes scalar arithmetic and bootstrap only. Original hashes of omitted records are commitments, not independently inspectable proof of their contents. See [README](../README.md) for the precise public/original receipt distinction and full-rerun boundary.
