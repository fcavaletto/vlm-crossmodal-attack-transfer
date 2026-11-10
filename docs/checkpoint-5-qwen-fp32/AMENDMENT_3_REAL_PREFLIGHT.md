# Checkpoint 5, amendment 3: real preflight and verified resume (2026-11-10, before the second RTX PRO 6000 attempt)

**Infrastructure and engineering only.** Approved by the author on 2026-11-10 after the first RTX PRO 6000 attempt. In that attempt every canary probe passed, but the first real model command failed on a CUDA-only bug in the memory guard ([`DEPLOYMENT_STATUS.md`](DEPLOYMENT_STATUS.md), fourth attempt).

Nothing scientific changes, and no CUDA result exists yet:
- the hypotheses, readings A–D, metrics and analysis script;
- the 48 development scenes and their order, the seeds, and the attack settings (ε = 1/255, FGSM, PGD-3, matched masks);
- gates G1–G9 and every threshold.

## Why

A canary made of small synthetic probes does not exercise the production code paths. The preflight must run those paths themselves on the real GPU, and do it before the operator's go.

## What changes

1. **Preflight = the production commands, unchanged in content, run before the go.** In order:
   - G3 input hashes;
   - the bf16 and the fp32 dtype audits (G1, G2, G8: real model loading in each precision, scoring, gradients, determinism);
   - clean scoring of all 96 prompts in both precisions (G5, G6);
   - finite differences in fp32 (G7);
   - `gates.py`;
   - **one complete paired unit**: the first scene of the frozen order, through the exact main runner (`run main --max-scenes 1`): bf16 then fp32, PGD-3, FGSM and the matched-mask scoring of `scene_block`. G4, G9 and the throughput gate are evaluated on it.
2. **Memory and runtime accounting.** The audit, clean and FD records now include peak CUDA allocated and reserved memory, host memory, disk, model load seconds and elapsed seconds. The paired unit records its seconds per precision and its peak reserved memory. It also records a projection made from its measured time: cost to completion, and finish time against the 4 h workload deadline.
3. **Preflight bundle and short hold.**
   - Everything except tensors is sent off the running pod. The tensors' hashes are in the integrity manifest.
   - I verify the sha256 locally, then check the gate results, memory and projections, **not the unit's scientific outcome**.
   - The pod then holds for at most **15 min**, polling every 60 s for `go <bundle sha256>`. A timeout or any other signal ends the workload, and the watchdog stops the pod.
4. **Verified resume.** After the go, `run main --resume` continues from scene 2. The preflight unit counts toward the 48 scenes only if:
   - `integrity.json` holds per-precision record hashes, scene ids and tensor hashes, written after each paired scene;
   - the files on disk still match it exactly;
   - both precisions completed the same leading scenes of the frozen order;
   - the scene order and seed prefix are unchanged.

   Otherwise the resume is refused and nothing is overwritten. Seeds depend only on the scene id, so a resumed run computes the same values as an uninterrupted one. The end-of-run drift check compares scene 1's clean score from the preflight process with a fresh score in the resumed process.
5. **G9 and throughput re-checked at resume**, using the preflight's measured time per paired scene and the real clock after the hold. Either gate failing stops the run before scene 2.
6. **Regression tests for the real entry points** ([`tests/qwen_precision/test_cuda_entrypoints.py`](../../tests/qwen_precision/test_cuda_entrypoints.py)).
   - They call `run.main()` with `--device cuda` for `audit` (bf16, fp32), `clean`, `fd` and `main`, through the real `make_guard` and `CudaGuard`.
   - A `torch.cuda` shim applies torch's device-index rule.
   - They cover: preflight then resume with each scene exactly once; resume refused on tampered records or without a preflight; a second fresh start refused; the throughput gate; and G9 at resume.
   - Against the guard that broke the fourth attempt, 10 of the 11 fail.
   - The model work is stubbed. On CUDA hardware, the preflight itself is the end-to-end test.
7. **The export manifest** is written inside the sent folder and printed to the pod log.

## Unchanged infrastructure

- RTX PRO 6000, Secure Cloud, at most $2.49/h, one create attempt.
- Absolute deadline 4.5 h; workload deadline 4 h; G9 at $12; $20 ceiling.
- Stop-first watchdog v2 (sha256 `a9ad711e…`); temporary pod storage; no network volume.
- Checkpoints during the run; export verified while the pod runs.
