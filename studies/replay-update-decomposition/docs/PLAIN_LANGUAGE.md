# What changed when the replay update was taken apart?

Imagine a network learning new clothes categories while rehearsing a few examples from older categories. A replay selector chooses which old examples to rehearse. High-loss chooses examples the network currently finds difficult. MIR chooses examples whose loss is predicted to increase after a trial fresh-learning step. Here that prediction is a signed change, not an absolute error or a guarantee of protection.

An earlier diagnostic found that the comparison between these selectors changed when the update recipe changed. This follow-up asks a narrower question: which parts of that recipe are involved?

## Two switches, four recipes

The first switch changes how strongly the new and rehearsed examples contribute. The deployed joint update averages sixteen fresh and ten replay examples together: the two contributions have weights 16/26 and 10/26. The alternative gives each group a full weight of one. That changes both their balance and the overall update size, so it cannot tell us which of those two ingredients matters separately.

The second switch changes where the replay correction is calculated: at the original network, or after the appropriately weighted fresh correction. Every recipe starts at the same saved network and uses exactly the same selected examples. No recipe trains a lasting replacement network.

Against high-loss replay, either single change leaves the average old-class MIR comparison favorable. Changing both yields an unfavorable average comparison. The switches therefore do not simply add up: their effects depend on one another. The reported +0.517077 interaction is that dependence, expressed in the study's loss units. It is not a newly identified biological mechanism.

## The warning matters more than the attractive relative comparison

MIR's deployed joint update still increases old-class loss by +0.905519 nats compared with doing nothing at the saved state. That damage occurs in every one of the sixteen states. Being slightly less harmful than another selector is not the same as preventing forgetting.

The joint comparison is favorable to MIR in only nine states. Its interval does not clear the study's chosen 0.005-nat practical margin. One of the four seed averages fails to reproduce the earlier endpoint sign reversal. Those limitations are part of the result, not failed rows to omit.

There is also a new-learning tradeoff. Compared with giving fresh examples their full update, joint MIR learns the new classes less well on average. Compared with the smaller fresh contribution built into the joint recipe, it has a favorable mean. Those are different questions, so presenting just one would be misleading. The comparison with high-loss does not establish a consistent new-learning advantage either.

## How much should we believe this generalizes?

These are sixteen checkpoints from only four historical training streams, reusing an already-used dataset and validation examples. The four streams, not the sixteen checkpoints, are the uncertainty units. The intervals describe variation within this tiny historical set; they are not a guarantee for new networks, new data or long-term memory.

The study clarifies this particular one-step computation. It does not show that MIR robustly works, that forgetting has been solved, that the best deployment recipe has been found, or that these computations map onto parts of a brain.

The public verifier checks the saved numbers and their arithmetic. It does not rerun the network or independently observe the original execution. See [the full evidence guide](../README.md) and [all signed results](RESULTS.md) for the details and missing-input boundary.
