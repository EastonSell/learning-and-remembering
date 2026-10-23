# Official TRAIN-only acquisition

Required source: zalandoresearch/fashion-mnist, pinned revision b2617bb6d3ffa2e429640350f613e3291e10b141.

Only these two compressed files are required:

- train-images-idx3-ubyte.gz, 26,421,880 bytes. SHA256 3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84. Published MD5 8d4fb7e6c68d591d4c3dfef9ec88bf0d.
- train-labels-idx1-ubyte.gz, 29,515 bytes. SHA256 a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845. Published MD5 25c81989df183df01b3e8a0aad5dffbe.

Create a new, empty directory outside this bundle, then download directly from the official repository:

Run this entire subshell. Directory creation is atomic and every output is exclusive; failure cannot fall through to another download. A failed or interrupted download leaves partial bytes for inspection, never an automatic retry or overwrite.

    (
    # BEGIN FAIL-CLOSED TRAIN ACQUISITION
    set -u
    set -C
    train=/absolute/TRAIN
    mkdir -- "$train" || exit 1
    curl --fail --location --proto '=https' --tlsv1.2 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/data/fashion/train-images-idx3-ubyte.gz' > "$train/train-images-idx3-ubyte.gz" || exit 1
    curl --fail --location --proto '=https' --tlsv1.2 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/data/fashion/train-labels-idx1-ubyte.gz' > "$train/train-labels-idx1-ubyte.gz" || exit 1
    python -B verify_inputs.py --training-assets "$train" || exit 1
    # END FAIL-CLOSED TRAIN ACQUISITION
    )

The integrity command checks exact size and SHA256 without decompressing. The shell noclobber setting protects redirections; mkdir failure stops the block before curl is invoked. The source URLs, size, SHA256, MD5 and Git blob identities are also machine-readable in input-catalog.json.

During bundle preparation the official pinned upstream [tree](https://github.com/zalandoresearch/fashion-mnist/tree/b2617bb6d3ffa2e429640350f613e3291e10b141/data/fashion), [README checksums](https://github.com/zalandoresearch/fashion-mnist/blob/b2617bb6d3ffa2e429640350f613e3291e10b141/README.md), and [MIT license](https://github.com/zalandoresearch/fashion-mnist/blob/b2617bb6d3ffa2e429640350f613e3291e10b141/LICENSE) were verified through read-only GitHub metadata/text. Existing compressed bytes were checked by hashes against those identities; no new dataset download or decode was performed. This verifies pinned source/integrity instructions, not a numerical rerun.

No TEST downloads, all-data preparation, historical training run, private evaluation lock, credentials or review authorization files are needed. Exact prepared training/validation IDs and duplicate-hash metadata are included. Their scientific preprocessing remains 28x28 uint8, nonoverlapping 2x2 float32 mean, division by float32(255), row-major 196 features.

Fashion-MNIST carries the upstream MIT notice, copyright 2017 Zalando SE, retained at inputs/data/LICENSE-Fashion-MNIST. Preserve it when redistributing dataset-derived material. The two raw TRAIN assets are deliberately not redistributed in this candidate.

