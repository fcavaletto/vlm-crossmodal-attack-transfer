# Checkpoint 1.8 review packet

DEVELOPMENT ONLY. Research lead: Federico Cavaletto. Qwen3-VL-2B-Instruct, one MPS load, no attack, no H* fit, not merged.

The measurement code is commit `fa7523ca9654beade6598449d4069995c37f10b2`. The scene ledger and the aggregate summary are commit `de364ac451cfa41c2df33106645aca8ddbba00c9`. Pull request: https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/18. It is not merged. The branch starts at `b9a31a80533bfa52c973a92c06364a4d74b4e537`, not at `main`, so the pull request also shows earlier checkpoints.

## What a labeled-color gradient is

The loss is \(L(y)\), the summed teacher-forced negative log-likelihood of the scene's own color. On this revision that color is one bare token: red 1151, blue 12203, green 13250, yellow 27869. The answer prefix is `<|im_end|>\n<|im_start|>assistant\n`, so the token is read from the first continuation logit. The mean over tokens is not the loss. Token 2518, the leading-space piece ` red`, is not the continuation.

The gradient is the derivative of that scalar with respect to the clean image as canonical float RGB in \([0, 1]\), one number per channel at each pixel. It is chained through the pinned resize, normalization, and patch layout. It is the local slope of the labeled-color loss at the unperturbed image. A positive component means increasing that channel at that pixel raises the loss, to first order.

## What was measured

Ninety-six `development_v4` scenes, under `neutral` and `helpful_nonrevealing`. One hundred ninety-two gradients. Each is finite and nonzero, with shape `(3, 384, 512)`. None is missing. The model was `Qwen/Qwen3-VL-2B-Instruct` at revision `89644892e4d85e24eaac8bacfd4f463576704203`, bfloat16, eager attention, loaded once after the existing budget selected bfloat16. Float32 did not clear that budget. SmolVLM2 was not loaded.

The four-scene gate passed. The scenes were the lexicographically first scene of each labeled color, chosen without looking at a gradient: `dev4-ref-00-a` (red), `dev4-ref-00-b` (blue), `dev4-ref-01-b` (green), `dev4-ref-14-b` (yellow). All eight gate gradients are finite and nonzero.

On all 96 images, `smart_resize` keeps 384×512. The differentiable pixels match the official PIL processor with maximum absolute gap 0. There is no scene outside the precommitted `1e-3` match.

The live summed NLLs match the stored Checkpoint 1.7 Qwen rows on all 192 primary pairs, with maximum absolute gap 0. The closed-vocabulary decision matches on all 192. The three loss formulas on the differentiated logit row agreed within `1e-5` on every scene. The Checkpoint 1.7 prediction file was not rewritten.

| Direction | Defined | Missing or undefined | Mean | Std | Min | p10 | p50 | p90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| \(A(\text{neutral}, \text{helpful})\) | 96 | 0 | 0.831 | 0.130 | 0.197 | 0.704 | 0.856 | 0.938 | 0.978 |
| \(A(\text{helpful}, \text{neutral})\) | 96 | 0 | 0.831 | 0.118 | 0.253 | 0.723 | 0.862 | 0.941 | 0.978 |

Both directions are positive on every scene. For \(A(\text{neutral}, \text{helpful})\), the bins of width 0.2 from −1 to 1 hold 0, 0, 0, 0, 0, 1, 1, 4, 16, and 74 scenes. The last bin is \([0.8, 1]\). For the opposite direction the counts are 0, 0, 0, 0, 0, 0, 1, 5, 21, and 69. Undefined alignments were not filled in as zero. There were none to fill.

| Context | L1 median | L1 mean | L1 max | L2 median | L2 mean | L2 max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| neutral | 549.1 | 4864.7 | 72511.6 | 3.16 | 24.12 | 341.7 |
| helpful_nonrevealing | 296.7 | 1934.6 | 54025.1 | 1.65 | 10.37 | 296.1 |

The means sit far above the medians. A few scenes have very large norms. The median is the more representative scale.

The clean margin is the best other color's summed NLL minus \(L(y)\). A positive margin means the labeled color has the strictly lowest sum. Neutral margins: mean 7.13, median 7.00, minimum −5.25, maximum 15.25, with 91 positive and 5 negative. Helpful margins: mean 6.81, median 7.00, minimum −1.25, maximum 15.50, with 95 positive and 1 negative. The sign of the margin matches the stored closed-vocabulary correctness on every row. Those counts are the Checkpoint 1.7 scores, 91/96 and 95/96, read again from this forward. They were not used to change the rule.

Repeats of the eight gate gradients matched exactly: relative L2 was 0 and the stability check confirmed on all eight. An appended-token forward on the same eight pairs stayed inside the documented bfloat16 relative ceiling of 0.5.

The directional finite-difference check did not. It uses the precommitted Rademacher direction, seed 17, steps \(1/64\) and \(1/128\), and the bfloat16 ceiling 0.5. One of the eight gate pairs was confirmed, `dev4-ref-00-a` under `helpful_nonrevealing`, by the smaller step (relative error 0.182). The larger step on that same gradient pointed the other way. The other seven pairs had no step inside the ceiling. Several smaller steps still had the analytic sign and still missed 0.5. This is a numerical failure of that check. The gradients themselves are not invalid: each is finite, nonzero, and stable under a repeat.

Wall clock was 4341.5 seconds, of which the load was 5.448 seconds. OS-available memory entering the budget decision was 12,813,271,040 bytes. The bfloat16 line is 11,509,447,200 bytes, so the load was permitted. The sampled minimum of OS-available memory after that was 6,483,738,624 bytes, which is above the 6 GiB floor of 6,442,450,944 bytes. Peak process RSS was 600.7 in the snapshot's mebibyte field. MPS current allocation stayed at 4072.9 in that same field. The driver allocation peaked at 5741.7.

Forty-eight of 48 counterfactual groups meet the prospective eligibility rule. All 96 scenes have finite, strictly positive L1 norms on both primary contexts and a finite alignment in \([-1, 1]\). Ninety-one scenes are also strict-correct under both contexts, which is the Checkpoint 1.7 clean-both count. Every group contains at least one of those 91, so the eligible-group count is 48. A missing gradient did not remove a group, because no gradient was missing. H* was not fit.

## What this does not show

These numbers are not an attack success rate and not a perturbation. No pixel was stepped, clipped, or projected. `project_linf` was not called.

\(A(s, t)\) is the share of the target gradient's L1 norm that lies along the sign of the source gradient. A value near 1 means the two clean-image gradients agree on the pixels that dominate the target. It is not evidence that a finite image step built under one context makes the labeled color wrong under the other. The finite-difference result is the local warning: a step of \(1/64\) at every pixel is already outside the neighborhood where the directional derivative predicts the loss. The association H* would test was not estimated. Alignment was not regressed on an attack outcome. There is no attack outcome.

Forty-eight eligible groups are the pre-attack count the Checkpoint 1.7 plan asked for. They do not replace that plan's attack rows or its cluster interval. Agent D's fabricated coverage note was not refit. The no-go for a confirmatory claim from this scene set remains a statement about missing attack outcomes, not about missing gradients.

The sample is the development set. Qwen's clean margins are large on most scenes, and the loss being differentiated is usually the loss of a color the model already assigns almost all of its probability. A gradient of a confident correct answer is a different object from a gradient near a tie. The norms span more than three orders of magnitude, so a single mean alignment does not describe every scene equally.

The bfloat16 load matches the budget rule and the Checkpoint 1.7 scoring dtype. Transformers did not set `config._commit_hash` on the loaded object. The call used the pinned revision, the token ids matched the protocol table, and the losses matched the stored rows exactly. That is the revision check this run has.

## Files

Compact records, not the gradient arrays:

- `docs/checkpoint-1.8/qwen_rgb_loss/summary.json`, sha256 `ced8e33fe1e305fc386661b26fa682d594bcadaae54d292c7828102f4288b15e`
- `docs/checkpoint-1.8/qwen_rgb_loss/progress.json`, one record per scene, with L1, L2, both alignments, the margin, and the float32 SHA-256 of each gradient
- `docs/checkpoint-1.8/qwen_rgb_loss/processor_gaps.json`, the processor gap on every scene

The Checkpoint 1.7 report's prose now states the 6 GiB floor as 6,442,450,944 bytes. The stored byte counts in that report, including 6373998592, are unchanged. That count is below the floor. The extra contexts in the pilot still ran because a later reading, after the primary pass, met the floor.

## Reproduction

From the worktree root. The unit tests do not load Qwen. The measurement command is the one that did.

```bash
HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/qwen_rgb_loss:src/qwen_rgb:src/choice_eval:src/vlm_iface:src/pilot:src/dev_pilot:src/analysis:src/scene_v4 \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pytest tests/qwen_rgb_loss -q

HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/qwen_rgb_loss:src/qwen_rgb:src/choice_eval:src/vlm_iface:src/pilot:src/dev_pilot:src/analysis:src/scene_v4 \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m qwen_rgb_loss
```

The second command loads Qwen once if the saved run is not already terminal. A saved budget refusal or an out-of-memory stop is not retried.
