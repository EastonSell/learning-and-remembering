# Prospective data amendment v1

Dated 2026-10-22. Frozen before dataset preparation, any model development,
validation outcomes, and any official-test predictions. This is an executable
clarification of protocol §1's required feature-grouped amendment if the original
stratified train-validation split leaks exact raw or transformed features.

## Deterministic grouping and allocation

1. Read all 60,000 official training images and labels. In ascending class order,
use one NumPy Generator(PCG64(73102)) to permute each class's official indices.
Retain the original first 1,000/class training and next 500/class validation indices.
2. Hash each raw uint8 image and each prescribed float32 transformed image using
SHA-256 over contiguous row-major little-endian bytes. Raw equality necessarily
implies transformed equality, so transformed equality defines grouping for the
training-validation allocation. No fitted transform or approximate matching.
3. Audit both raw and transformed overlap. If neither overlaps, retain the
original allocation without amendment.
4. If overlap exists, keep the exact original 1,000/class training indices. Assign
every full-training-set feature group represented there to the training partition,
including groups with conflicting labels. For each class in ascending order,
scan its same original permutation from the start and select the first 500 indices
whose transformed feature group is not training-owned. All remaining items are
ignored. A feature group's selected members may occur only in one partition;
unused members need not be selected. Repeated features within one partition are
not deduplicated. This keeps exact 1,000/500 class quotas without splitting groups.
5. Re-audit both raw and transformed equality and stop if overlap remains or a
fixed class quota cannot be met. Save original and final actual index arrays,
per-item feature hashes, NumPy version, and file/payload SHA-256 hashes.

This procedure uses labels only for the already-prespecified stratified quotas,
not validation model outcomes. It changes validation membership only when
necessary and makes no person/writer-disjointness claim.

## Official-test audit, no predictions

Compare every official-test raw and transformed hash with the union of selected
training and validation hashes. Exclude a test item if its raw OR transformed
hash matches. The exclusion function receives no labels. Only after fixing this
mask, count retained labels and require at least 500/class. A failure stops the
confirmatory path, with an integrity manifest preserving the failure. Do not tune
or relax the rule. Report both retained-primary and full-official descriptive
metrics only after a complete lock and separate authorized scoring stage.

Data files may be integrity-checked now; no prediction, loss, accuracy or per-item
error is computed by preparation. Prior use of the official test for project
method selection remains unresolved until explicitly attested in the complete
lock. A prepared manifest is not a lock and never grants scoring permission.

## Storage and immutability

Pinned source: zalandoresearch/fashion-mnist revision
b2617bb6d3ffa2e429640350f613e3291e10b141, HTTPS raw repository data/fashion files and
LICENSE only. Compare all four published compressed-file MD5 values and record
independent SHA-256 hashes; inspect IDX magic, lengths, dimensions and class counts.
Retain the official MIT notice. Dataset bytes and preparation arrays live outside
the code/public root. New artifacts use exclusive creation and refuse overwrite;
an existing raw file is only reused after its published integrity check passes.
The preparation manifest records this amendment's SHA-256.
