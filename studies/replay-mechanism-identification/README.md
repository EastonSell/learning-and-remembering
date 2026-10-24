# Replay mechanism identification supplement

This additive, post-hoc supplement reanalyzes the retained scalar outcomes of the [historical 2×2 update decomposition](../replay-update-decomposition/README.md) and supplies an exact quadratic counterexample. It does not run a neural model, establish the trained model's mechanism, demonstrate novelty, or add independent empirical evidence.

Start with [MECHANISM.md](MECHANISM.md) for the signed results, identification limit, and proposed fixed-coordinate transport/deformation diagnostic. The proposal remains unrun. A null deformation loss attribution does not imply absence of differential parameter drift.

## Reproduce without model execution

From this directory, using Python's standard library only:

```sh
python -B -S analyze.py
python -B -S -m unittest -v test_analysis
```

The analysis regenerates `saved_results.json`, `state_results.csv`, `source_identities.json`, and `synthetic_results.json` in place. The ten tests cover the analytic controls. Saved empirical results and synthetic examples remain separate. All sixteen retained state files represent four historical seed clusters, not sixteen independent trials.

`MANIFEST.json` preserves the reviewed source-file SHA256 identities. Its own SHA256 is `f97ac56ec3a6c9bddd23564ee0dbe8e6ed0c4cb8faa07f00b0f7a74aed321b49`. This README is publication packaging, outside that frozen source manifest. The frozen report's statements about no remote writes refer to the original analysis phase, before this additive publication. `inputs/publication_receipt.json` describes the earlier saved-arithmetic publication, not this supplement.

Ten tests and byte-identical regeneration of all four outputs were independently checked before publication. These checks do not constitute full-repository validation, fresh model evidence, or confirmation of the proposed mechanism. Historical source files are preserved unchanged.
