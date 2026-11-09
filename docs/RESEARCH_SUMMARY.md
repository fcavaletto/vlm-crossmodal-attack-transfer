# Research contribution summary

Federico Cavaletto · for PhD and AI-safety research applications · November 2026

## Short version (about 120 words)

I showed, with a pre-registered held-out confirmation, that 3-step PGD's large advantage over FGSM on a vision-language model (Qwen3-VL-2B, ε = 1/255) comes from **which** RGB channel coordinates it perturbs, not how many. Under 8-bit rounding, PGD-3 is exactly a subset of FGSM's perturbation, which allows a clean matched-count decomposition. On 48 newly generated scene groups:
- PGD's subset beat same-size random subsets by +6.50 logits (95% CI 6.08–6.92);
- sparsity explained 3% of the gap;
- coordinates ranked by clean-gradient magnitude did worse than random.

For safety evaluation, this means single-step attacks and gradient saliency can misjudge both how vulnerable a model is and where. I built the whole pipeline on one laptop, including a bit-exact reproduction harness and hash-frozen protocols.

## Longer version (about 300 words)

**Problem.** Cheap, gradient-local tools are used to estimate how vulnerable a model is and which inputs matter. They include single-step attacks, saliency maps and first-order damage estimates. All of them assume the model is locally close to linear.

**Finding.** On a 2B vision-language model at one 8-bit level per channel, 3-step PGD does about 3× FGSM's damage. I proved that after the official 8-bit rounding, the PGD-3 image is FGSM's own perturbation restricted to about 30% of its coordinates, with identical signs. That made a decomposition possible with no confounds: same values, same signs, same count, different coordinates.
- A pre-registered development control found that coordinate *selection* explains nearly all of PGD's advantage.
- I then built an independently seeded scene generator with provable separation from all development data, hash-froze the protocol and analysis, and ran a single blind confirmation.
- All three predictions held. Selection was +6.50 logits [6.08, 6.92]. The sparsity share was 3% [−2%, 9%]. Top-|clean gradient| selection was worse than random (−1.38 [−1.71, −1.06]).
- The clean first-order model ranks the subsets backwards.

**Rigor.**
- Every run reproduces earlier records bit-for-bit.
- Memory-safety policies were fixed before the runs.
- Deviations are recorded, and earlier results are annotated rather than edited.
- An earlier pre-registered hypothesis (H\*) that could not be estimated is reported as it stands, not reframed.

**Limits.** One model, one synthetic task family, one ε, one precision. The mechanism is not established, and only two clean-gradient rules were tested.

**Next.** A development-only pilot tests whether gradient sign *consensus* at random nearby inputs, sampled independently of the attack, reproduces PGD's selection. It includes controls for bf16 numerics, 8-bit quantization and spatial clustering, and compares methods at equal gradient-evaluation budgets. If it holds, it suggests a principled fix for cheap evaluations ("vote before you rank") and a parallel one-shot attack. Either would need its own held-out confirmation.

## Skills this demonstrates

- adversarial ML on multimodal models;
- pre-registration and held-out confirmation design;
- reproducible numerical engineering (bf16 and MPS determinism, 8-bit quantization);
- honest reporting of null and stopped results.

Repository: <https://github.com/fcavaletto/vlm-crossmodal-attack-transfer>. Manuscript: [`docs/paper/MANUSCRIPT.md`](paper/MANUSCRIPT.md).
