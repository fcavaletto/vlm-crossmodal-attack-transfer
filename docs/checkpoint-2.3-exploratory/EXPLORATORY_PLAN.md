# Checkpoint 2.3 exploratory plan: perturbation paths, precision, and direction (2026-11-09)

Status: **exploratory, development data only.** Approved by research lead Federico Cavaletto on 2026-11-09 (decision S3, including the fp32 investigation and the reverse-direction attack). This file is committed **before** any measurement in it is taken.

## 0. Rules

- **Data.** Only `development_v4` scenes are used. No sealed data, no v5 scene, and no new sample.
- **Compute.** Mac only (MPS). No paid infrastructure.
- **Frozen results.** The Checkpoint 2.1 attack records (`docs/checkpoint-2.1/development/`) are not edited, rerun in place, or overwritten. New outputs go only to `docs/checkpoint-2.3-exploratory/`.
- **Status of everything new.** Every attack variant, measurement, and hypothesis here is exploratory. Nothing here tests H\* or H-ΔL, or changes either protocol.
- **Guards.**
  - The frozen memory guard is reused unchanged: the Checkpoint 1.5 budget, the 6 GiB OS floor, one `from_pretrained` per process, and refusal when another model process is resident.
  - The frozen attack constants are reused unchanged: ε = 1/255, step ε/4, clean start, sign steps on the labeled-color summed NLL, the 8-bit round trip for official scores, and the Checkpoint 2.0 random-sign seed rule.
- **Time cap.** Total model time is ≤ 4.6 h, enforced in code (§5). If the cap binds, the run stops starting new work and the report says what was not measured. The run is never silently extended.

## 1. Central question

How do small image perturbations (L∞ = 1/255) produce such different effects across:
- **attack methods**: FGSM, PGD-3, PGD-7, and random sign, which all end on the same 1/255 ball after rounding;
- **textual contexts**: `neutral` and `helpful_nonrevealing`, as source and as target?

How much of the difference is explained by each of these?
- **(E1) Softmax curvature:** NLL is nearly exponential in the logit margin.
- **(E2) Change of the gradient along the path:** rotation, magnitude change, sign flips.
- **(E3) Numerical precision:** bf16 inputs, bf16 activations, and bf16 logits.
- **(E4) The 8-bit round trip.**

None of these is assumed. In particular, the plan does **not** assume that first-order approximations fail. It measures first-order predictions and realized changes side by side, in NLL and in margin, at every step.

The question is **not** answered from attack-success differences. Every claim about geometry must rest on measured gradients, perturbation vectors, or loss trajectories.

## 2. Scenes (fixed now, outcome-blind)

- **Main set, S48:** arm `a` of all 48 development groups, `dev4-ref-00-a` … `dev4-ref-47-a`. One scene per group, so scenes are group-independent.
- **Subset, S16:** the 16 S48 scenes whose `counterfactual_group_id` has the smallest `SHA-256("checkpoint-2.3-exploratory-subset|" + group_id)`. The rule reads no outcome. It yields: `dev4-ref-00-a, 03, 07, 12, 16, 20, 25, 27, 29, 30, 33, 37, 38, 39, 42, 47-a`.
- **Order:** S16 first, in group order, then the other 32 in group order. If the cap binds, the scenes lost are main-set-only scenes, never S16.
- **Clean-incorrect scenes** under either context are still measured. Analyses report them separately, and none is dropped silently.

## 3. Precision modes

| Mode | Vision tower | Language model | Logits | Where |
| --- | --- | --- | --- | --- |
| **B** (frozen) | bf16 (pixel inputs cast to bf16) | bf16 | bf16 | every measurement |
| **B+P** | bf16 | bf16 | **fp32 projection**: the final bf16 hidden state times the lm_head weights, both in fp32 | recorded alongside every B forward pass, free |
| **V32** | **fp32** (pixels stay fp32) | bf16 | bf16, plus the fp32 projection | S16, stage 3 only |
| Full fp32 | fp32 | fp32 | fp32 | **not possible on this machine** |

Why full fp32 is out:
- The frozen guard requires 20.87 GB free for an fp32 load.
- The Mac has 24 GB in total, and Checkpoint 2.1's bf16 run already went as low as 7.3 GB free.
- The guard is not lowered.

Why each partial mode helps:
- B+P separates **output quantization**, the 0.125-step logits, from everything upstream.
- V32 separates **input and vision-tower quantization** from the language model's.
- Neither removes bf16 inside the language model. The report must say so.

V32 runs only if a memory check passes right before conversion:

> available bytes ≥ 6 GiB + 2 × (vision parameter bytes in fp32) + 1 GiB

If the check fails, V32 is skipped and the report records the limitation. It is not retried with a lower guard.

## 4. Measurements

Notation:
- `x` is the clean image; `δ` is a perturbation in RGB [0, 1] units.
- `g_c^L` is the RGB gradient of context c's labeled-color summed NLL. This is the frozen objective, computed with the frozen `backward_at_image`.
- `g_c^M` is the RGB gradient of context c's margin, `z_label − z_comp`, where `comp` is the best competitor at the clean image.
- "Float path" is the differentiable pixel path, with no rounding. "Official" is the 8-bit processor path.

### 4.0 Model-free, before loading weights (Q0, all 48 scenes)

For the processor's normalized pixel tensor:
- the fraction of coordinates whose **bf16** value changes when x moves by `t·ε·s`, for t ∈ {1/8, 1/4, 1/2, 3/4, 1} and a fixed random sign vector s;
- the relative error `‖bf16(x+δ) − bf16(x) − (x+δ − x)‖₂ / ‖x+δ − x‖₂`, in normalized units.

This measures how much of a perturbation survives the bf16 input cast. No model is needed.

### 4.1 Clean (mode B, S48)

- **Official scores**, both contexts: NLL, margin, and decision in B and B+P.
- **Float-path scores**, both contexts, computed **twice**. This is a determinism check.
- **Gradients** `g_n^L, g_h^L, g_n^M, g_h^M`: four backward passes. For each, record ‖g‖₁ and ‖g‖₂.
- **Pairwise:**
  - `cos(g_n^L, g_h^L)` (never stored before);
  - `cos(g_c^L, g_c^M)`;
  - A(n,h) and A(h,n);
  - sign agreement between `g_n^L` and `g_h^L`.

### 4.2 Per direction (S48), source s ∈ {n, h}, target t the other context

These are matched: the same constants, the same steps, and the same scorer for both directions.

- **FGSM:** `δ_F = Π(ε · sign(g_s^L))`.
- **Random sign:** matched support.
  - Neutral source: the frozen seed `scene_seed("checkpoint-2.0-calibration", scene_id)`.
  - Helpful source: `scene_seed("checkpoint-2.3-exploratory|helpful", scene_id)`.
- **Line scan** along `sign(g_s^L)` on the float path, t ∈ {1/8, 1/4, 1/2, 3/4} (t = 1 is FGSM).
- **PGD-3:** frozen update, 3 steps. S16 continues the same trajectory to 7 steps.
- **At every evaluated point** (each line-scan point, FGSM, random, and each PGD iterate δ_k):
  - **realized:** float-path NLL and margin under **both** contexts, in B and B+P;
  - **clean-gradient first-order predictions**, both contexts: `g_c^L·δ` (NLL) and `g_c^M·δ` (margin);
  - **per-step local prediction** (PGD only): `g_{s,k−1}^L·(δ_k − δ_{k−1})` against the realized source NLL change of that step;
  - **perturbation geometry:** ‖δ‖₁, ‖δ‖₂, ‖δ‖∞; the fraction of coordinates at ±ε, and at 0; `cos(δ, δ_F)`; and the sign disagreement with δ_F among coordinates where both are nonzero.
- **Gradients along PGD:** at each iterate where one is computed (k = 1, 2, and on S16 also 3–6):
  - ‖g_k‖₁, ‖g_k‖₂;
  - `cos(g_k, g_0)`, `cos(g_k, g_{k−1})`;
  - sign agreement with g_0, unweighted and |g_0|-weighted.
- **Official (8-bit round trip)** at FGSM, random, PGD-3, and (S16) PGD-7, both contexts, B and B+P. The rounded-δ geometry is recorded as above.
- **Reproduction check:** neutral-source official NLL, margin, and decision for FGSM, random, PGD-3, and PGD-7 are compared with the frozen Checkpoint 2.1 records, which are read only. Any mismatch is reported, not corrected.

### 4.3 Stage 3: V32 (S16, neutral source only)

1. Convert the vision tower to fp32, if the memory check passes.
2. Repeat 4.1 for `g_n^L, g_n^M, g_h^L`, and 4.2 for the neutral source with PGD-3, both contexts, all in V32.
3. Cross-evaluate:
   - the B-crafted perturbations (stage 1 FGSM, PGD-3, and PGD-7, kept as int8 multiples of ε/4) under V32;
   - the V32 perturbations under B, after converting the vision tower back to bf16. This is an exact inverse: the fp32 weights are an exact upcast.
4. Gradient agreement: `cos(g_n^L[B], g_n^L[V32])` and sign agreement.

### 4.4 Storage

- Scalars go to `docs/checkpoint-2.3-exploratory/paths/records.jsonl`, with run metadata in `progress.json`. Both are committed.
- Gradient tensors (fp16) and perturbations (int8) go to `paths/tensors/`. They are **not** committed (about 0.5 GB). The report says how to regenerate them.

## 5. Budget

The plan caps model time at 4.6 h, from measured costs: one backward ≈ 17 s, one forward ≈ 1 s (Checkpoint 2.1).

| Stage | Work | Estimate |
| --- | --- | --- |
| Load + Q0 | — | about 5 min |
| Stage 1 (S48) | 8 backward + about 60 forward passes per scene | about 3.3 min per scene, about 2.6 h |
| Stage 2 (S16 extra, inline) | steps 4–7, both directions: 8 backward + about 20 forward passes | about 2.6 min per scene, about 0.7 h |
| Stage 3 (S16, V32) | about 5 backward + about 40 forward passes, plus cross-evaluation | about 2.3 min per scene, about 0.6 h |
| **Total** | — | **about 3.9 h** (cap 4.6 h) |

Enforcement:
- Stages 1 and 2 start no new scene once elapsed model time plus the running mean scene time would exceed 3.9 h.
- Stage 3 starts only if elapsed model time is < 4.0 h, and starts no new scene past 4.6 h.
- Unmeasured scenes are listed in `progress.json`.

## 6. Analyses (all descriptive)

Intervals are percentile bootstraps over scenes (S48: one scene per group, so this is a group bootstrap), 9,999 draws, seed 20261109. No p-value decides anything.

1. **First-order accuracy by method and context.** Realized / predicted, in NLL and in margin, with the exact margin gradient. Reported as medians and interquartile ranges, and as log-log slopes. B against B+P.
2. **Where the method difference comes from:**
   - `g_s^L·δ` for FGSM against PGD-3 and PGD-7 (by construction FGSM maximizes it);
   - realized changes for each;
   - geometry: the sign-disagreement fraction, cos(δ_PGD, δ_F), and the moved fraction;
   - gradient rotation per step;
   - per-step local first-order accuracy.

   The claim "the clean linear model misranks perturbations" is made only if realized and predicted orderings are measured to disagree.
3. **Line scans.** The shape of ΔM(t) and ΔL(t): linear, concave, convex, or stepped. They are compared against the Q0 bf16-survival fractions at the same t.
4. **Precision:**
   - the determinism check;
   - B against B+P margins: is quantization at 0.125 the main error?
   - B against V32: the first-order ratio, the line-scan shape, the random-sign noise floor, gradient agreement, and cross-evaluated damage.

   Stated as limits: what V32 cannot rule out (LM bf16).
5. **Direction.** For n→h and h→n separately:
   - **source effectiveness:** source margin drop and NLL rise, and the source flip rate;
   - **conditional transfer:** target flips among source flips;
   - **continuous target damage:** target margin drop and NLL rise for all scenes;
   - the target drop per unit source drop, along the path, at **matched source margin drop** (linear interpolation over the PGD iterates);
   - the first-order predicted ratio, A·‖g‖₁, as a baseline.
6. **Reproduction:** the agreement rate with the frozen 2.1 records.

## 7. What this can and cannot show

**Can:**
- whether realized changes track first-order predictions, per method, context, and precision mode;
- whether FGSM and PGD differ in direction, in magnitude, or both;
- whether gradients rotate along the path;
- how much the output and vision precisions matter;
- whether the transfer direction matters at matched source damage, on these 48 scenes.

**Cannot:**
- establish a mechanism inside the language model, or rule out bf16 effects inside it;
- generalize beyond Qwen3-VL-2B, these synthetic scenes, and these two contexts;
- confirm any hypothesis. Findings are hypothesis-generating and need a fresh split (`study2-trackB-confirm-v1`, not drawn).
