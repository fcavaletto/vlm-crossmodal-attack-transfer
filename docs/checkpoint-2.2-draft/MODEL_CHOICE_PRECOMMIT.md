# Checkpoint 2.2 model choice, committed before the development Q analysis

Written 2026-11-08. It is committed **before** `mechanistic_predictor.py` is written or run, and before anyone relates Q, A, or any gradient quantity to a development outcome. The git history is the evidence for that order.

What had been looked at by this point:
- the 2.1 development `ΔL_h` SD (2.96), skew (0.66), and sibling correlation (0.48);
- the mean FGSM `ΔL_h` as a fraction of its first-order prediction (about 0.08);
- the 2.1 binary counts.

No regression of a development outcome on a predictor has been run.

## Facts that fix the choice (no outcome needed)

1. Checkpoint 1.8's `A(neutral, helpful)` is `g_hᵀ sign(g_n) / ‖g_h‖₁` (`src/analysis/analysis/metrics.py:182`). So the requested predictor is `Q = g_hᵀ sign(g_n) = A · ‖g_h‖₁`. It is computable exactly from the stored scalars.
2. The first-order prediction for three sign steps of ε/4 along a fixed `sign(g_n)` is `P_h = 0.75 · ε · Q`. The previous draft's model (b) term `A · S_h` is exactly `P_h`, so model (b) already is the Q model.
3. `Q = A · ‖g_h‖₁` splits into magnitude and direction. Any information Q carries beyond the gradient magnitude comes from `A`. A test of "Q beyond baselines" is a test of the direction, as long as the magnitude baselines are flexible enough. If they are not, a nonlinear magnitude effect can masquerade as a Q effect, which is the main validity threat.
4. `P_h` spans about two orders of magnitude, and realized changes are far below first-order at this ε (step 2 above). So a linear-in-`P_h` model gives the largest-gradient scenes most of the leverage, exactly where the linear picture is worst.

## Primary model (M1, saturating first-order)

`ΔL_h = β0 + β_Q · f(P_h) + γ1 · f(S_h) + γ2 · M_n + γ3 · M_h + γ4 · log‖g_n‖₁ + γ5 · log‖g_h‖₁ + e`

where:
- `f(P) = c · asinh(P / c)`. It equals `P` for `|P| ≪ c` (the first-order regime) and grows like `c · log P` for `P ≫ c` (saturation). It is odd, so negative predictions keep their sign.
- `S_h = 0.75 · ε · ‖g_h‖₁` is the same prediction at perfect alignment. `f(S_h)` is in the model so the magnitude enters in the same nonlinear form as `f(P_h)`. Then `β_Q` is identified only by `A`, which addresses fact 3.
- `H-ΔL: β_Q > 0`, one-sided, α = 0.025. There is one primary and no split.

**How c is fixed (decided now, applied once).** `c` is calibrated on development data from the **source side only**: the neutral self-attack, which does not involve the helpful outcome.
- Fit `ΔL_n(PGD-3) = a + b · c · asinh(0.75 · ε · ‖g_n‖₁ / c)` on the 96 development scenes, clustered by group.
- Grid: `c ∈ {0.25, 0.5, 1, 2, 4, 8, 16, 32}` nats. Choose the `c` with the smallest residual sum of squares.
- If the minimum lies on a grid end, take that end and say so.
- The chosen `c` is frozen into the protocol. The sealed data never re-chooses it, except in a labeled sensitivity.

Assumption, stated rather than tested: the response function that maps first-order predictions to realized changes is the same under both texts. The same image and the same attack path are being read through two prompts.

## Secondary model (M2, additive sign-alignment)

`ΔL_h = β0 + β_A · A + γ2 · M_n + γ3 · M_h + γ4 · log‖g_n‖₁ + γ5 · log‖g_h‖₁ + e`

This is the earlier draft's model (a). It is agnostic about mechanism, and it is what a "does alignment predict" reader would fit first.

## Baselines (nested, all reported)

| Name | Columns |
| --- | --- |
| B0 | margins `M_n`, `M_h` |
| B1 | B0 + `log‖g_n‖₁`, `log‖g_h‖₁`, `log‖g_n‖₂`, `log‖g_h‖₂` |
| B2 | B1 + `f(S_h)` |
| B3 | B2 + `cos(g_n, g_h)` (sealed data only; development did not store it) |

Reported for each, with group-bootstrap intervals and grouped cross-validated R²:
- `f(P_h)` added to B2;
- `A` added to B2;
- `cos` added to B2;
- `f(P_h)` added to B3 ("Q beyond cosine").

Descriptive only, no gate.

## How the development Q analysis may be used

It may report how accurate first-order is (source side and transfer side, FGSM and PGD-3), Q's incremental association, and the error structure for the power simulation. It may **not** change M1, M2, the baselines, or the rule for `c`. If it suggests a different model, that goes into the review packet as a labeled observation, and the user decides. Development estimates of `β_Q` are not used to set the effect size for the sample-size calculation.
