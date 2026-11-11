# Research strategy after Checkpoint 2.2: two tracks (DRAFT, 2026-11-09)

> **Historical record (note added 2026-11-09).** This is the strategy proposal as written on 2026-11-09, kept unedited below. It is not the current plan. Track B of this proposal became Checkpoints 2.3 and 2.4; the current roadmap is [`docs/research/ROADMAP.md`](../research/ROADMAP.md).

For research lead Federico Cavaletto. This is a proposal. Nothing in it has run, and it needs approval (§6).

Since the last review:
- no model was loaded;
- no scene was generated or rendered;
- no pod was created;
- nothing was merged.

The only new computation is `track_b_existing.py` (`.json`). It reads committed development records, and it was written after every number it reads was known. Its results are **exploratory and post hoc**. They are used here only to decide which experiments are worth running, never to choose a confirmatory specification.

> **H\* and H-ΔL stay separate.** H\* is the proposal's original binary-transfer hypothesis. It stopped at Checkpoint 2.1 (`effective_sample`, 4 transfer failures against a floor of 40), and that result stands. H-ΔL was formulated on 2026-11-08, after the H\* counts were seen, and it was never preregistered. Nothing below edits any H\* document or the precommit file (`MODEL_CHOICE_PRECOMMIT.md`, `d2321f7`). Track B's three hypotheses are newer still (2026-11-09), and they are exploratory on `development_v4`.

## 0. Summary

1. **Track A is not worth 320 groups now.**
   - The development point estimate is r̂ ≈ 0.44 of full transfer, with a 95% WCU interval from −0.53 to 1.41.
   - At r = 0.5, 320 groups gives power 0.30, and even 512 groups gives only 0.44.
   - Even full transfer (r = 1) would explain about 1.3% of the outcome variance.
   - Recommendation: freeze the corrected protocol (M1s), run no confirmatory sample before 16 November, and reframe any later run as an **estimation study** that can rule out full transfer (§1.3).
2. **The development data already reframes opportunity 1.** "First-order overestimates damage" is true for FGSM but looked false for PGD (PGD-3 realizes about 1.0–1.3× in NLL). In margin space, which removes the softmax curvature, both fall far short of first order: FGSM realizes about 2.5% and PGD-3 about 11%. The question is not "why FGSM", but why a 1/255 ball is so nonlinear, and why three gradient steps find a vertex of the **same** ball that drops the margin 2.75× more than FGSM does.
3. **Gradient magnitude mostly measures confidence.** Across scenes, log ‖g_n‖₁ against clean margin has Spearman −0.85. The NLL gradient scales with 1 − p. This is a likely reason the magnitude terms dominated H-ΔL.
4. **Ranking (§5):**
   1. Opportunity 1, local nonlinearity;
   2. Opportunity 3, asymmetry. It shares opportunity 1's harness.
   3. Opportunity 2, text defenses. It has the highest safety relevance but the highest risk, and it should start after 16 November with a preregistered prompt set.
5. **Recommendation for 16 November:**
   - one combined exploratory run on development data, covering opportunities 1 and 3: Mac only, $0, about 5 h at 48 scenes;
   - a short write-up that turns the H-ΔL null into a mechanistic finding;
   - Track A presented as a finished, reviewable design.

## 1. Track A: confirmatory H-ΔL

### 1.1 Primary model: M1s, and why M1 was inadequate

The precommitted M1 is a saturating first-order model in `P_h = 0.75 ε Q`, with **linear** controls for log ‖g_n‖₁ and log ‖g_h‖₁. In simulation (`spline_check.py`), the world was given:
- no alignment effect;
- a smooth nonlinear magnitude effect, quadratic or tanh in log ‖g_h‖₁, at 1 nat per SD.

M1's WCR test then rejected **0.106–0.118** of the time at nominal 0.025.

The reason: `f(P_h)` and `f(S_h)` correlate at 0.992. After the linear controls, the residualized SD of `f(P_h)` is 0.04. So `f(P_h)` is close to a nonlinear function of the magnitudes, and it absorbs any curvature the linear controls miss. The test then attributes magnitude curvature to alignment.

M1s adds restricted cubic splines (4 knots, at the 5/35/65/95 percentiles of the predictor pool) in both log norms. It holds size at 0.023–0.027 in those worlds, and power under the M1 world is unchanged (0.709 against 0.71).

Retained as proposed primary (decision D1). The departure from the precommit is:
- justified by simulation alone, with no development outcome used to choose it;
- recorded in `PROTOCOL_DRAFT.md` and `REVIEW_PACKET.md`.

The precommit file is not edited.

New supporting fact (part 6 below): gradient L1 tracks clean confidence, with Spearman −0.85 against margin. The magnitude controls are therefore partly confidence controls, which is a further reason to model them flexibly.

### 1.2 Is 320 groups worth it?

| Quantity | Value | Source |
| --- | --- | --- |
| Dev M1 coefficient | 2.50 (WCU 95% −3.03 to 8.04) | `mechanistic_predictor.json` |
| In units of full transfer, r = β / 5.70 | **r̂ = 0.44** (−0.53 to 1.41) | same |
| WCR power at r = 1 | 0.42 / 0.72 / 0.82 / 0.86 at G = 128 / 256 / 320 / 384 | `sample_size.json` |
| WCR power at r = 0.5 | 0.15 / 0.26 / 0.30 / 0.39 / 0.44 at G = 128 / 256 / 320 / 384 / 512 | same |
| Partial R² at r = 1 | 0.013 | same |
| Runtime at 320 groups | about 21 h on the Mac (31 h with S1), plus 1–2 days to build and test the v5 generator | `REVIEW_PACKET.md` §6 |
| Cost | $0 in compute, about 4 calendar days | — |

Assessment:
- The study is powered only for the optimistic end of the dev interval.
- Near the dev point estimate, a 320-group test most likely ends "not significant", which is uninformative about r ≤ 0.5.
- A positive result would be an effect of about 1% of the variance, in a synthetic task, through a predictor whose first-order basis is weak (§2.1).

Spending a week of the remaining time on it would buy little before 16 November, and the time has a better use (§4).

### 1.3 Recommended reframing (decision S2)

Keep the M1s test, but make its primary deliverable the **estimate of r with its interval**. The study then answers a question for any true value: can full first-order transfer be ruled out?

Rough projection:
- The dev CR1 standard error is 3.64 in β, which is 0.64 in r, at 46 groups.
- Scaling by √(46/G) gives about 0.27 at 256 groups and about 0.24 at 320.
- The upper 97.5% bound would then be about r̂ + 0.5. If r̂ stays near 0.44, 256 groups would exclude r = 1.

**Caveat:** this projection uses CR1, which over-rejects, and dev-v4 alignment spread. The real figure comes from `sample_size.py` rerun on dev-v5 predictors (G1). The reframing must be fixed before any sealed outcome exists.

Options:
- **(a) Recommended: freeze and defer.** Freeze PROTOCOL_DRAFT with M1s, and run nothing confirmatory before 16 November. Later, run at `G_final = 256` as an estimation study. The test is still reported, with its power stated in advance.
- (b) Run the 320-group test as drafted (D3).
- (c) Retire H-ΔL after the feasibility stage if G1 shows that v5 does not spread alignment more than v4.

### 1.4 Feasibility stage that cannot touch the sealed sample

| Stage | Data | What it may change | What it may not |
| --- | --- | --- | --- |
| F0, done | `development_v4` scalars and records | Nothing confirmatory. It motivates S2. | — |
| F1 = gate G1 | dev-v5, 32 groups, seed `study2-checkpoint-2.2-dev-v5`. Clean scoring and gradients only, **no attack**, about 50 min. | `G_final`, through alignment spread, by the predictors-only rule in `V5_GENERATOR_SPEC.md` §5 | Model form, link, or covariates (frozen before F1) |
| F2, optional pilot | **pilot-v5**, a new split with seed `study2-checkpoint-2.2-pilot-v1`, 24 groups, with attacks | Runtime, failure rate, and error SD for the power rerun only. Every pilot result is reported. | Model form; the effect-size anchor; whether to report. Never pooled with sealed. |

Safeguards:
- The novelty rule (`V5_GENERATOR_SPEC.md` §3) is extended so sealed groups exclude pilot-v5 layouts and mirrors.
- The F2 seed string is new, so drawing it changes no other split.
- The go or no-go rule for the sealed run is fixed before F2 and uses only precision: proceed if the projected half-width of the r interval at `G_max` is ≤ 0.6. It never uses the pilot β.

## 2. Track B: what the existing records already show

Source: `track_b_existing.py`, all 96 complete development scenes. This is a post-hoc description, not a test.

**1. Clamping is ruled out.** Every rendered pixel channel lies in [32, 236]. The renderer draws without antialiasing, so a ±1/255 step can never hit the [0, 1] box.

**2. The output side has a precision floor.** All 1,536 stored margins are multiples of 0.125, and 91% are multiples of 0.25. These are bf16 logits. The smallest nonzero margin change is 0.125, and 4.3% of (scene, attack, context) cells show no margin change at all.

**3. Same ball, very different damage.** After the 8-bit round trip, FGSM, PGD-3, and PGD-7 all have L∞ = 1/255. Before the round trip, PGD-3 was at 0.75/255.

| Attack | Neutral NLL rise (median) | Neutral margin drop | Neutral flips | Helpful flips (transfer) |
| --- | --- | --- | --- | --- |
| Random sign | −0.001 | −0.19 | 2% | 0% |
| FGSM | 0.13 | 3.1 | 19% | 18% |
| PGD-3 | 3.61 | 9.5 | 57% | 65% |
| PGD-7 | 10.67 | 17.0 | 88% | 91% |

Paired by scene, PGD-3's margin drop is **2.75×** FGSM's (p10–p90 1.9–5.9×).

**4. First order, NLL against margin.** The margin prediction is approximate (`ε‖g_NLL‖₁/(1−p)`, which assumes one dominant competitor).

| | Realized / predicted, NLL | Realized / predicted, margin (approx.) |
| --- | --- | --- |
| FGSM | 0.067 | **0.025** |
| PGD-3 | 0.96 | **0.11** |

The NLL is roughly exponential in the margin, which makes PGD look first-order-accurate in NLL. In margin space, both are an order of magnitude short of linear.

**5. Damage barely tracks gradient size.**
- ‖g_n‖₁ spans about 120× between p10 and p90 (73 to 8,900). The PGD-3 margin drop spans 3.3× (4.3 to 14.3).
- Spearman of margin drop against ‖g_n‖₁ is 0.26 for FGSM, 0.18 for PGD-3, and −0.17 for PGD-7.
- log ‖g_n‖₁ against clean margin has Spearman −0.85.

**6. Alignment is symmetric; gradient norms are not.**
- A(n,h) median 0.856 and A(h,n) median 0.862, correlated at 0.93.
- The helpful gradient is smaller: the median of log(‖g_h‖₁/‖g_n‖₁) is −0.53, so ‖g_h‖₁ is about 0.59× ‖g_n‖₁.
- First order therefore predicts n→h transfer at about 0.57× h→n (p10–p90 0.17–1.7).
- **No helpful-source attack has ever been run**, so the realized asymmetry is unknown.

**7. No sign that helpful text protects.**
- Clean margins are the same under both contexts (median 7.0, helpful above neutral in exactly 50% of scenes).
- The transferred PGD-3 flips helpful at 65%, against 57% for neutral under direct attack.
- The transfer ratio ΔL_h/ΔL_n rises with attack strength: 0.61 for FGSM, 0.66 for PGD-3, 0.82 for PGD-7.
- None of this measures robustness, because no attack targeted the helpful loss.

## 3. Track B: three opportunities

Shared rules for all three:
- Data is `development_v4` only. It is already used for H-ΔL exploration, so anything found here is **hypothesis-generating**.
- Confirmation needs a separate fresh split, seed `study2-trackB-confirm-v1`. It is excluded from, and never pooled with, sealed-v5 for H-ΔL.
- Each run's measurement list and analysis plan is committed before the run.

### 3.1 Opportunity 1: why first-order predictions fail inside a 1/255 ball

**Hypotheses.** These are stated so they can each fail. They are not mutually exclusive, and the experiment estimates each one's share.
- **H1a, softmax saturation.** The FGSM shortfall in NLL is mostly the softmax: the NLL gradient is scaled by 1 − p ≈ 0.004 at clean. *Prediction:* with the **margin** gradient, FGSM realizes most of its first-order margin prediction. Part 4 above already points against this (approx. 2.5%), so H1a is expected to fail, and testing it properly is cheap.
- **H1b, gradient rotation.** The margin gradient rotates within the ball. *Prediction:* cos(g(x + tδ), g(x)) and the sign agreement between steps fall sharply for t ≪ 1. The extra damage at PGD step k comes from directions absent in g₀. The line scan ΔM(t) along sign(g₀) is concave and flattens well before t = 1.
- **H1c, numerical floor.** bf16 forward precision makes the loss a staircase at this scale. In normalized pixel units (std about 0.27), 1/255 is about 0.015, which is 2–4 bf16 ulps for values of magnitude 0.5–2. *Prediction:* in fp32, FGSM's realized/predicted ratio rises materially, and the line scan becomes smooth.

**Why it matters for safety.**
- Robustness evaluations and transfer predictors (gradient alignment, FGSM-based screening, linear-probe arguments) assume local linearity. If a 1/255 ball is strongly nonlinear in a deployed-precision VLM, single-step evaluations understate vulnerability by about 3× here.
- Gradient-geometry predictors of transfer, the premise of H\* and H-ΔL, would then need multi-step quantities.
- If H1c holds, robustness numbers depend on inference precision, which matters for how safety evaluations are run.

**Minimum informative experiment, E-B1.**
- 48 scenes, arm `a` of each group, so groups are independent.
- Neutral source, bf16 as frozen.
- Per scene:
  1. The clean **margin** gradient (true answer logit minus the best competitor), alongside the NLL gradient.
  2. A line scan along sign(g₀) at t ∈ {1/8, 1/4, 1/2, 3/4, 1} · ε, recording margin and NLL under both contexts.
  3. PGD-7 with per-step logging: the source margin and NLL; ‖g_k‖₁ and ‖g_k‖₂; cos(g_k, g₀); sign agreement with g₀; the change in each context's margin; and the fraction of pixels moved after the round trip.
  4. The Hamming distance between the FGSM sign pattern and the rounded PGD-3 and PGD-7 perturbations.
  5. A two-point directional curvature: (g(x + h·s) − g(x))·s / h along s = sign(g₀).
- fp32 control (H1c): the same, minus PGD-7, on 16 of the 48 scenes, run in fp32 if the memory budget check in `qwen_rgb/budget.py` allows it.

**Compute.** All on the Mac, $0.
- About 23 s per PGD record and about 21 s per clean gradient have been measured.
- The plan is about 2.5 min per bf16 scene (about 2 h for 48 scenes) and about 6 min per fp32 scene (about 1.6 h for 16).

**Implementation.** About 1 day:
- a logging wrapper around `attack_feasibility.steps.pgd_update`;
- a margin-loss scorer;
- a line-scan driver;
- tests with a toy model, so no weights are needed;
- one-scene smoke, then the run.

**Strongest alternative explanation and controls.**
- *The round trip, not the geometry.* Rounding adds or removes moves, and PGD-3 goes from 0.75/255 to 1/255. Control: record every quantity before and after the round trip.
- *Saturation.* Control: report in margin space throughout.
- *Precision* is itself H1c, tested with the fp32 control.
- *Selection.* Control: report all 48 scenes, with no eligibility filter beyond clean correctness.
- *A trivial result.* "PGD beats FGSM" is known. Control: the analysis is about **why**, decomposed per scene into saturation, rotation, and precision shares.

**Novel versus known.**
- Known: PGD is stronger than FGSM (Madry et al. 2018). Local nonlinearity within ε-balls is documented for image classifiers: local-linearity regularization (Qin et al. 2019); gradient alignment and catastrophic overfitting (Andriushchenko & Flammarion 2020; Wong et al. 2020).
- New here:
  - the measurement in a VLM, whose output is a text-conditioned token logit;
  - whether local linearity **differs across textual contexts** on the same image;
  - the bf16 precision floor as a separable component;
  - an explanation of why first-order transfer predictors fail for multi-step attacks.
- The last item turns the H-ΔL null into a finding.
- Publishable as a workshop paper if the decomposition is clean and replicates on a second model. A single model on synthetic scenes is a strong application piece, not yet a paper.
- *Citations are from memory and must be checked before any write-up.*

### 3.2 Opportunity 2: can text reduce vulnerability without causing answer copying?

**Hypothesis H2.** There is a non-revealing context C such that **adaptive** attacks (PGD on C's own loss) at ε = 1/255 succeed less often than under neutral, while:
- image dependence is preserved: counterfactual-arm accuracy stays as high, and accuracy collapses when the color evidence is ablated;
- reliance on text does not rise: under C plus a contradictory sentence, the model does not follow the sentence more often.

**Why it matters for safety.**
- System-prompt defenses are cheap and widely deployed.
- If one works only because the model stops looking at the image, or because the gradients get harder to use, it is a false sense of security.
- The counterfactual-pair design here can tell those apart. Most VLM robustness benchmarks cannot.

**Existing evidence (§2, part 7).** None favorable. The helpful text leaves clean margins unchanged and flips at least as often under transferred attack.

**Minimum informative experiment, E-B2.**
- 24 groups (48 scenes, both arms, because image dependence is measured across arms).
- Contexts:
  - the four existing contexts;
  - **two** new candidates, written and committed before any run: for example, an instruction to attend to color evidence, and a robustness warning.
- Per context:
  - adaptive PGD-3 and PGD-7;
  - FGSM;
  - the random-sign control.
- Image-dependence probes, forward-only:
  - both arms correct;
  - a grayscale or color-shuffled image, where accuracy should drop to chance;
  - the contradictory-sentence follow rate.

**Compute.**
- About 48 × 6 × (2 × 23 + 4) s ≈ 4 h on the Mac, plus about 0.5 h of forward-only probes. $0.
- Implementation about 1.5 days: context registry, probes, and the adaptive-attack loop. Design effort is the largest part.

**Strongest alternative explanation and controls.**
- *Gradient masking or obfuscation*, where the context makes the gradients unhelpful rather than the model robust (Athalye et al. 2018; Tramèr et al. 2020). Controls:
  - success must rise monotonically with steps;
  - FGSM must not beat PGD;
  - success at a large ε must reach about 100%;
  - a transfer attack from neutral must not beat the adaptive one.
- *Answer copying.* Controls: the revealing context as the copying ceiling; the ablation and contradictory probes.
- *Forking paths*, from trying prompts until one works. Control: the candidate list is fixed in advance, and every candidate is reported.

**Novel versus known.**
- Known: prompt-based defenses for multimodal jailbreaks exist (for example, adaptive shield prompting, 2024 — *verify*). The rule that defenses must survive adaptive attacks is established.
- New here: a controlled test that separates **robustness** from **not looking at the image**, using counterfactual arms and ablation. A clean null ("text defenses don't confer image-grounded robustness at 1/255") is useful and modest. A positive result that survives all four masking checks would be surprising and publishable.
- Highest safety relevance, and the most room for a false positive.

### 3.3 Opportunity 3: is transfer asymmetric between neutral and helpful?

**Hypothesis H3.** Transfer efficiency, the target margin drop per unit of source margin drop, differs between n→h and h→n. It differs **beyond** what first order predicts from A and the gradient norms, at matched source damage.

**Why it matters for safety.**
- If attacks crafted under a generic prompt transfer to protective or task-specific prompts as well as the reverse, an attacker does not need to know the deployed system prompt.
- Asymmetry shows where the vulnerability sits:
  - symmetric transfer points to shared visual features;
  - asymmetric transfer points to a context-specific readout.

**Existing evidence (§2, part 6).** Alignment is symmetric (corr 0.93). Norms are not, so first order predicts n→h at about 0.57× h→n. The realized reverse direction has never been measured.

**Minimum informative experiment, E-B3.**
- The same 48 scenes and logging as E-B1, with **helpful** as the source: PGD-7 plus FGSM.
- Because every step is logged, transfer can be compared at **matched source margin drop**, by interpolating along each path. That removes the "stronger source attack" confound without retuning ε.

**Compute.** About 1.2 min per scene beyond E-B1 (clean gradients are shared), so about 1 h. $0. Implementation: about 2 hours on top of E-B1, since the source context is already a parameter.

**Strongest alternative explanation and controls.**
- *Magnitude and saturation.* Helpful's smaller gradient makes it harder to move. Controls: matched source damage; margin space; the first-order prediction as an explicit baseline.
- *Ceiling effects at PGD-7.* Control: analyze at steps 1–3 as well.
- *Scene heterogeneity.* Control: a paired within-scene comparison, reported with the group bootstrap.

**Novel versus known.**
- Known: cross-prompt transferability of adversarial images on VLMs has been studied (for example, cross-prompt attacks, ICLR 2024 — *verify*).
- New here: the **directional** comparison at matched source damage, explained by measured gradient geometry, in a counterfactual testbed.
- Moderate novelty. Strongest as a section of the opportunity 1 paper, not alone.

## 4. Recommendation: what to execute before 16 November

Today is 9 November, so seven days remain.

| Day | Work | Model use | Cost |
| --- | --- | --- | --- |
| 9–10 Nov | Approve S1–S6. Commit the E-B1 and E-B3 measurement and analysis plan, marked exploratory. Freeze PROTOCOL_DRAFT with M1s and record its sha256 (Track A done as a design). | none | $0 |
| 10–11 Nov | Implement the path logging, margin scorer, and line scan. Toy-model tests. One-scene smoke on the Mac. | 1 scene | $0 |
| 11–12 Nov (overnight) | Run E-B1 (bf16, 48 scenes) + E-B3 (helpful source), about 3 h. Then the fp32 control (16 scenes), about 1.6 h. | Mac MPS | $0 |
| 13–14 Nov | Analysis: decompose the shortfall into saturation, rotation, and precision shares; compare asymmetry at matched damage. Figures. | none | $0 |
| 15 Nov | A 4–6-page write-up (below), plus a buffer. | none | $0 |
| 16 Nov | Submit. | — | — |

The write-up covers:
- **H\***: the preregistered result, stopped at `effective_sample`, reported as is.
- **H-ΔL**: the post-hoc hypothesis; the dev null on Q beyond magnitude; the M1 → M1s correction and the simulation behind it; the frozen, powered design.
- **E-B1 and E-B3**: why first-order fails, labeled exploratory.

That story shows preregistration discipline, a self-found statistical flaw, and a mechanistic follow-up, which is what a safety research application rewards.

Not before 16 November:
- Opportunity 2;
- the v5 generator, G1, and the pilot;
- any CUDA pod.

CUDA is not needed for any of the work above. The $3 validation plan stays ready.

**Risks.** If E-B1 slips, the write-up still stands on §2. Those are committed, post-hoc numbers, presented as such.

### After 16 November: continuing the strongest direction

1. **Weeks 1–2:**
   - replicate E-B1 on fresh scenes (`study2-trackB-confirm-v1`) with its hypotheses fixed from the dev run;
   - add a second VLM (SmolVLM2, already supported) to test generality.
2. **Weeks 2–4:** opportunity 2, with the E-B1 tools:
   - a preregistered candidate prompt list;
   - adaptive attacks;
   - the four masking checks;
   - image-dependence probes.

   This is the direction with the clearest safety story, and E-B1's measurements are what make its masking checks credible.
3. **Track A, when time allows:** build v5, run G1 and the optional pilot, and then run the sealed estimation study at 256 groups (§1.3), or retire H-ΔL by rule (c).
4. **Target:** a workshop paper on local nonlinearity and transfer in VLMs, combining opportunities 1 and 3 across two models, with opportunity 2 as the main paper if its result survives the masking checks.

## 5. Ranking

| | Scientific interest | Feasibility by 16 Nov | Safety relevance | Application value | Overall |
| --- | --- | --- | --- | --- | --- |
| **1. Local nonlinearity (E-B1)** | High: §2 already shows a large, unexplained effect | High: development data, about 1 day of code, about 2 h of compute | Medium: evaluation validity | High: turns the null into a mechanism | **1st** |
| **3. Asymmetry (E-B3)** | Medium | High: shares E-B1, about 1 h | Medium–high: attacker knowledge of the prompt | Medium | **2nd** |
| **2. Text defenses (E-B2)** | High if positive, modest if null | Low: design-heavy, high false-positive risk | **High** | High, but only if done carefully | **3rd now; lead candidate after 16 Nov** |

## 6. Decisions requiring approval

| # | Decision | Recommended | Alternative |
| --- | --- | --- | --- |
| S1 | Two-track strategy; Track A frozen with M1s (D1) and **no** confirmatory run before 16 Nov | Yes | Run 320 groups now (D3 as drafted) |
| S2 | Track A primary deliverable | An estimate of r with an interval, plus the M1s test, at `G_final = 256`. Fixed before any sealed outcome. Supersedes D3 and D5. | The D3 rule with `G_max` 320 |
| S3 | Exploratory run E-B1 + E-B3 on `development_v4`, 48 scenes (arm a), Mac only, about 3 h | Approve. This lifts "no model inference" for this dev-only run. | 96 scenes (about 6 h) |
| S4 | fp32 control on 16 scenes, if the memory budget check passes | Approve | Skip H1c |
| S5 | Opportunity 2 after 16 Nov, with a preregistered prompt list | Yes | Start before the deadline (not recommended) |
| S6 | New seed namespaces: `study2-checkpoint-2.2-pilot-v1` (F2) and `study2-trackB-confirm-v1`, both excluded from sealed-v5 by the novelty rule | Yes | — |

D1, D2, D4, and D6–D10 in `REVIEW_PACKET.md` stand as recommended. D3 and D5 are replaced by S2 if it is approved. G1–G4 and C0–C4 are unchanged and not needed before 16 November.
