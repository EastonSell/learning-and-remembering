# Replay interaction: a useful identification result and next experiment

## Findings worth acting on

1. **The saved interaction is not simply “relocating MIR makes old loss worse.”** At unit weights, relocating replay lowers mean old-panel loss by 0.427161 for MIR and 0.899276 for high-loss. Both selectors improve in 10/16 states. High-loss benefits more. At deployed weights the corresponding reductions are 0.542594 and 0.497631. The interaction is exactly the difference between these differential benefits: (−0.427161 + 0.542594) − (−0.899276 + 0.497631) = +0.517077. These are benefits relative to original-location replay, not absolute preservation: unit-updated MIR still raises old loss versus the original checkpoint by +1.202602 on average, and high-loss by +0.862479, with 16/16 states adverse for both. This is scalar reanalysis of saved outcomes, not a causal mechanism estimate.
2. **Magnitude is concentrated, but its sign is not a one-seed artifact.** Seed 2005/block 1 contributes 47.47% of the sum of state interactions; here MIR's unit-location change is adverse (+3.018057), unlike its overall mean. All four leave-one-seed-out interaction means stay positive: +0.266493 to +0.657206. These sensitivity descriptions do not add independent observations or justify a new significance claim. Removing the largest state produces an unbalanced descriptive mean +0.289738; it is not the primary seed-balanced estimand.
3. **A concrete identification counterexample:** smooth, convex, one-dimensional quadratic losses with identical replay Hessians and no representation changes reproduce D00<0, D01<0, D10<0, D11>0. Therefore those signs and positive interaction do not establish selector-specific replay curvature, neural representation change, or a universal advantage/disadvantage of MIR. This is an existence proof, not a fitted explanation of the saved model.

## Exact synthetic control

Use one parameter x initially zero, eta=3/10, a=8/13, b=5/13, and the same four update formulas as the public study. There are two candidate replay items and a quota of one:

- Fresh F(x)=(x+1)^2/2, with F'(0)=1.
- Replay R_M(x)=2(x−1/20)^2.
- Replay R_H(x)=2(x+1/20)^2+1.
- Independent old evaluator V(x)=(x+1/20)^2/2.

High-loss actually selects H, since its current loss is larger. MIR actually selects M, since its signed loss increase after the full fresh virtual step x=−3/10 is larger. Constants affect high-loss selection but not gradients; the curvature of both replay items is exactly 4. Select once and freeze across all cells. This is a two-item analytic selection control, not a ten-item or Fashion-MNIST replication.

Let alpha,beta be (a,b) or (1,1), and l be 0/1 for original/updated replay gradient location. Endpoints satisfy

x_M = −eta alpha + eta beta r + l eta^2 alpha beta h,
x_H = −eta alpha − eta beta r + l eta^2 alpha beta h,

where r=1/5 and h=4. Let their midpoint be m and separation d=x_M−x_H. Exactly,

D = V(x_M)−V(x_H) = d(m+1/20),
d=2 eta beta r,
m=−eta alpha + l eta^2 alpha beta h.

Thus location changes the common midpoint, not selector separation. The interaction is

I = 2 eta^3 r h (1−a b^2) > 0.

There is no selector Hessian difference at all. A common replay-gradient drift moves the pair through a curved evaluator and changes its loss ordering. The mechanism is evaluation geometry plus shared transport. It needs neither nonconvexity nor training histories.

The exact sign-reversal conditions in this family, with evaluator optimum t and r>0, are: both original-location midpoints lie below t, the deployed updated midpoint remains below t, and the unit updated midpoint crosses above t. For t=−1/20 and eta=.3 these hold whenever 25/9 < h < 455/72 (approximately 2.778<h<6.319). Our h=4 lies inside this open interval; equality yields a zero contrast rather than a strict reversal.

Nearby controls:

- Halving eta to .15, retaining h=4 and the same candidate selections, makes all four D negative. I remains positive but becomes exactly one eighth as large. A positive interaction alone therefore does not require a sign reversal.
- Replacing V with an affine evaluator makes location interaction exactly zero, despite nonzero shared gradient drift.
- Freezing replay gradients (h=0 in the endpoint control, not a claim about the original quadratic candidate losses) makes interaction exactly zero even with quadratic V.

Exact rational results and ten assertion tests are included. The literal endpoint test was corrected after it exposed a hand-calculation error in the test expectations; the final saved output reports all ten passing. No trained model or original prospective StageA experiment was run or reconstructed.

## Why the usual small-step explanation is incomplete

For a smooth replay loss R_s and frozen selector s, original-location endpoint q_s=theta−eta(alpha g_F+beta g_s). Moving replay evaluation to theta−eta alpha g_F shifts the endpoint by

u_s=−eta beta [g_s(theta−eta alpha g_F)−g_s(theta)]
   =eta^2 alpha beta H_s g_F + O(eta^3).

Consequently the selector-relative location effect is

D_w1−D_w0 = eta^2 alpha beta grad(V)(theta) dot (H_M−H_H)g_F + O(eta^3),

and I=eta^2(1−ab) grad(V)(theta) dot (H_M−H_H)g_F + O(eta^3).

This is a local expansion for smooth losses and fixed selected sets, not an estimate at eta=.3. ReLU boundary crossings would further qualify it. If H_M=H_H, or the evaluator gradient is near zero, the leading term can vanish while higher-order evaluation geometry remains. The exact toy is a cubic example. A claim that a positive I proves differential Hessian transport is therefore invalid. The saved scalar branch losses do not contain the parameter vectors or gradients needed to distinguish these alternatives; hashes cannot recover them.

## Strongest next step: fixed-set common-transport vs deformation intervention

This is a proposed new, explicitly admitted experiment on newly available states, NOT reconstruction or execution of lost StageA files. Do not launch it without the needed environment, inputs and authorization. It requires no eta sweep or selector reselection.

For each weight setting, retain the already-required endpoints q_M0,q_H0,q_M1,q_H1 from independently discarded branches. Define midpoints m_l=(q_Ml+q_Hl)/2 and differences d_l=q_Ml−q_Hl. Define C(m,d)=V(m+d/2)−V(m−d/2).

Existing observed contrasts are C00=C(m0,d0) and C11=C(m1,d1). Add only two hybrid contrasts:

- C10=C(m1,d0): transport the original selector pair to the new midpoint while preserving its difference.
- C01=C(m0,d1): change selector difference while preserving the original midpoint.

Each hybrid contrast requires evaluation at two parameter vectors. This is four extra panel-endpoint evaluations per weight setting, eight across the two weight settings per state. Both old/new panels can be scored at each endpoint. No extra gradient calls are required beyond the original branch computation. Keep all model buffers/evaluation state fixed; if the model has non-parameter state, specify and control it explicitly.

Use symmetric attribution, not a favorable ordering:

Transport A_w = [(C10−C00)+(C11−C01)]/2.
Deformation B_w = [(C01−C00)+(C11−C10)]/2.

Exactly A_w+B_w=D_w1−D_w0 and I=(A_1−A_0)+(B_1−B_0). The analytic control predicts B_w=0 and all interaction attributed to A. This attribution tests which endpoint change contributes to the evaluated loss contrast, not whether differential parameter drift exists. A claim that a material part of this evaluated interaction is attributable to changing selector separation is contradicted if the corresponding B_1−B_0 contribution is negligible under a predeclared margin. Conversely, substantial B falsifies an account attributing the entire evaluated location effect solely to common transport under this fixed-coordinate decomposition. Neither conclusion identifies a unique Hessian mechanism. Do not use a post-hoc “dominant” label: predeclare a numerical reconstruction tolerance and a materiality margin (e.g. existing .005-nat yardstick, explicitly exploratory), report both signed contributions, seed means, and every state, and use seed clusters as the uncertainty unit. Mixed or cancelling contributions are a valid result.

Report ||d_1−d_0|| (differential endpoint drift), ||m_1−m_0|| (common endpoint transport), and each ||u_s|| alongside the loss attribution, with a declared parameter scaling convention. Zero B does not imply d_1=d_0: evaluator-null directions, local symmetries, or cancellation can hide nonzero differential drift. Small B_1−B_0 also does not imply that either individual B_w is small. A scalar norm detects vector change but not its scientific relevance or a unique Hessian cause.

An exact two-dimensional null control makes the limitation concrete. Set q_M0=(0,0), q_H0=(1,0), q_M1=(0,1), q_H1=(1,−1), and V(x,y)=x^2/2. Then m_0=m_1=(1/2,0), while d_0=(−1,0), d_1=(−1,2): differential drift has norm 2. All four hybrid contrasts are −1/2, so A=B=0 despite nonzero differential drift. This is included in the test suite and synthetic result file. It prevents interpreting a null loss attribution as absence of differential replay-gradient movement.

This gives exact, falsifiable attribution in the model's fixed parameter coordinates, not an invariant biological/representational mechanism. Parameter midpoint interventions are coordinate-dependent and can be off the original update path. State this limitation. Do not infer persistent retention from one-step panels.

A separate later crossed-alpha/beta study would separate replay ratio from total update scale. It is not needed to answer this first mechanism question and must not be silently folded into the existing four-cell evidence.

## Reproducibility, sources and limits

Run from this directory:

    python -B -S analyze.py
    python -B -S -m unittest -v test_analysis

Only Python standard-library code is used. `inputs/evidence/` contains exact copies of 16 saved public state JSONs. Each is SHA256- and Git-blob-verified against the retained source manifest before analysis. `source_identities.json` lists identities. `state_results.csv`, `saved_results.json` and `synthetic_results.json` keep empirical saved arithmetic separate from synthetic results. No bootstrap, new neural training/inference, original checkpoint reconstruction, credentials, downloads, API calls, remote writes or repository modifications occurred in this task.

Sources: public EastonSell/learning-and-remembering, original inspected commit 69411a790bb59161fb5d44bdfeefb1c8d76669f2. Existing publication receipt records unchanged frozen study blobs at b0aeb63658c38b17e0e7ee42284dac724a75ee1c; this task makes no new live-head claim.

- Methods: https://github.com/EastonSell/learning-and-remembering/blob/b0aeb63658c38b17e0e7ee42284dac724a75ee1c/studies/replay-update-decomposition/docs/METHODS.md
- Exact inspected mechanics: https://github.com/EastonSell/learning-and-remembering/blob/b0aeb63658c38b17e0e7ee42284dac724a75ee1c/studies/replay-update-decomposition/reference/jointaudit/mechanics.py
- Raw state example: https://github.com/EastonSell/learning-and-remembering/blob/b0aeb63658c38b17e0e7ee42284dac724a75ee1c/studies/replay-update-decomposition/evidence/seed-2005-block-1.json

Scientific scope: four historical seed clusters, sixteen states; post-hoc explanatory work. No fresh causal evidence, fitted mechanism, confirmation, unseen generalization, or future retention claim. The toy demonstrates non-identification, not that this is what happened in the trained system.
