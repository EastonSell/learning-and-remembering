# Exact declared logical operation ledger v1

An operation is one scalar add, subtract, multiply, divide, exp, log or comparison
in the declared numerical graph. Dot(m,k,n)=m*n*(2*k-1). No fused-operation discount.
Functions in core.py are the executable convention; tests enforce cumulative caps.
This model is exact for that declared graph, not exact machine instructions/FLOPs.

For batch n, forward_ops counts:
- input-hidden dot, bias adds and ReLU comparisons;
- hidden-output dot and bias adds;
- per-row maximum comparisons, logit shifting, exponentials;
- TWO exponent sum reductions (the implementation intentionally computes both),
  softmax divisions, log sums and log-probability subtraction.
loss_ops adds n negations. gradient_ops adds label subtraction, normalization,
three gradient dot products, both bias reductions, ReLU mask comparisons and
multiplications. update_ops adds 2*6634 scalar operations for SGD. selection_ops(n)
is update_ops(16)+2*loss_ops(n)+n deltas. diagnostic_ops(n) adds update_ops(26).

All four diagnostic arms execute that graph every base update. Native efficient
uniform would execute only update_ops(26); high-loss would add one loss_ops(n);
MIR/shuffledscore retain full selection_ops(n). These are theoretical native costs,
not separate native-efficient training results. Score diagnostics, finite checks,
index/gather/copy/concatenation, RNG, sorting, JSON and I/O are NOT scalar numerical
kernel operations; their counts/bytes or measured runtime are separately disclosed.
The useful-compute control does not claim all-computer-cost matching: it matches
this numerical-kernel budget. Base candidate draws remain matched; extra sampler
and Python overhead can make wall time differ. No unmeasured speedup claim follows.

Useful control: cumulative allowance += diagnostic_ops(n); used += update_ops(26).
Spend floor(residual/update_ops(10)) extra ten-item replay-only steps, subtracting
each exact declared cost. Carry residual forward. Never borrow future allowance.
The first update has n=16; n subsequently reaches 50 as the reservoir fills.
Candidate evaluations count as memory-related forward computation even if unused.

Audit costs are outside learner allowance and explicitly logged: full-reservoir
virtual score, extra candidate correlation check, two ten-item replay branches,
and two old-validation loss passes. They occur equally often in every arm. Audit
candidate sorting and statistical reductions are reporting overhead. Scoring logs
five full forward passes plus argmax comparisons; recall aggregation is reporting
overhead. Training, auditing, scoring and total CPU/wall time must be distinguished.

Persistent raw parameter bytes: 26,536. Reservoir raw bytes: 158,600. A virtual copy
adds 26,536 transient bytes, with gradients, activations, sampled minibatches,
score vectors and sorting buffers in addition. Runtime records traced allocated
peak, sampled RSS, RNG-state serialization bytes and admission-counter object bytes.
These do not hide Python/native overhead behind parameter matching.

Full-schedule checksum constants: candidate sizes are sixteen for eight updates,
thirty-two for four, forty-eight for four, then fifty for 2,484. Each diagnostic
case uses 6,200,562,008 logical operations. The useful-compute case performs 15,853
extra replay-only updates, uses 6,200,428,622 operations, and leaves 133,386 unspent.
A base 26-item update costs 713,678; an extra ten-item update costs 278,574.
These constants are independently checked by the result verifier.
