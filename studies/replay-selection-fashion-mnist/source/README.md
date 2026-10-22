# Public replay-study source

This is the Python implementation used in the completed Fashion-MNIST replay-selection study. All Python source and tests are byte-for-byte frozen originals. Scientific docs are historical prospective documents, with two editorial changes cataloged in ../PUBLICATION-NOTES.json. This README replaces the private historical workflow README; do not claim the original whole-tree source freeze passes on this edited public documentation.

See ../README.md for results, the redacted historical-export verifier, environment requirements, and new-run instructions. A new run uses a NEW source freeze of this public tree. Never change the historical source-freeze-original.json or use old results as a new prospective experiment.

Tests use synthetic temporary fixtures only:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v

Fresh freeze and check (output must not exist):

    python run.py freeze --output /absolute/new-work/source-freeze.json
    python run.py verify-freeze --freeze /absolute/new-work/source-freeze.json

Scientific code and its safety/provenance gates are unchanged. An operator must obtain a genuine independent review of their new frozen protocol and record their own execution decision. The helper ../prepare_new_run.py creates a local clearance receipt only when explicitly invoked with that review and acknowledgement. It does not grant permission, perform review, train, score, download, or reuse any historical clearance.
