# Which Coordinates, Not How Many: Matched-Count Evidence on the PGD–FGSM Gap in a Vision–Language Model

**Pre-registered, held-out evidence that 3-step PGD's advantage over FGSM on a vision-language model comes from *which* RGB channel coordinates it perturbs, and that two standard clean-gradient rules rank those coordinates below random.**

Federico Cavaletto · 2026 · Qwen3-VL-2B-Instruct · L∞ ε = 1/255 · main study on an Apple-silicon laptop; fp32 check on one rented GPU

![Figure 1: matched-count decomposition of PGD-3 vs FGSM](docs/checkpoint-2.4-confirm/figures/figure1_selection.svg)

**Figure 1.** **(A)** Mean margin drop, with 95% group-bootstrap CIs, for four perturbations of the same image in the same ε-ball:
- dense FGSM;
- a random subset of FGSM's perturbation;
- the subset with the largest clean-gradient magnitude;
- the subset that PGD-3 keeps.

All three subsets have exactly the same number of coordinates and FGSM's own signs. Orange markers are the development data; blue markers are the pre-registered held-out confirmation. **(B)** One point per scene group. PGD-3's subset beats a same-size random subset in every group of both samples.

## What we found

At the smallest meaningful budget, one 8-bit level per channel, 3-step PGD does about **3× the damage** of FGSM on a 2B-parameter vision-language model. After the official 8-bit rounding, the PGD-3 image is FGSM restricted to about 30% of its RGB channel coordinates, with the same signs, apart from at most 7 zero-gradient rounding ties per image (out of about 175,000 kept coordinates). This follows from the arithmetic of three ε/4 sign steps (see [the report, §3](docs/checkpoint-2.4-confirm/REPORT.md#3-a-structural-fact-that-is-not-the-finding)). It allows a matched comparison: hold the perturbation values, signs and count fixed, and vary *which* coordinates are kept. It controls perturbation magnitude, sign and count. It does not separate a coordinate's individual effect from its spatial arrangement or from interactions among the kept coordinates.

On **48 newly generated scene groups**, analyzed under a protocol hash-frozen and pushed before any inference ([`PROTOCOL.md`](docs/checkpoint-2.4-confirm/PROTOCOL.md), commit `c35b02c`):

| Pre-registered prediction | Held-out result (95% CI) | |
|---|---|---|
| PGD-3's coordinates beat random coordinates at the same count | **+6.50 logits** margin damage [6.08, 6.92]; positive with either context as source; 96/96 group-by-source cases | supported |
| Sparsity explains at most 25% of the PGD-versus-FGSM gap | **3%** [−2%, 9%] | supported |
| Top-\|clean gradient\| coordinates are worse than random | **−1.38 logits** [−1.71, −1.06] | supported |

- On scenes the model answered correctly, PGD-3's subset changed the answer in 60–78% of cases, against 18–23% for random subsets of the same size.
- The selection effect carries almost undiminished to the context the image was not optimized for.
- The clean first-order prediction ranks the subsets backwards. It rates PGD's subset at 0.92× random and the top-gradient subset at 2.8× random. In reality PGD's subset does 2.8× random and the top-gradient subset does 0.6×.

**Why it matters for safety evaluation.** Single-step attacks, magnitude saliency and first-order damage estimates are the cheapest tools for estimating how vulnerable a model is and for locating which inputs matter. Here they misjudge both: they underestimate the vulnerability, and they point to the wrong coordinates. Iterative attacks, scored after quantization, are the stable measurement.

**What this does not show.** It does not show that the clean gradient carries *no* information about the damaging coordinates. Only two rules were tested, magnitude ranking and the first-order prediction. It also does not explain *why* PGD's coordinates are damaging, and it covers one model, one synthetic task family, one ε and one precision.

## Status of each claim

| | Claim | Where |
|---|---|---|
| **Confirmed** (pre-registered, held-out) | Selection, not sparsity, explains PGD-3's advantage. Top-\|∇\| selection is worse than random. | [Checkpoint 2.4 report](docs/checkpoint-2.4-confirm/REPORT.md) |
| **Structural** (arithmetic) | Under these settings the PGD-3 image is a sign-preserving FGSM subset. The only exceptions are 0–7 half-level ties per image, at coordinates with exactly zero gradient. | Report §3, §5.6 |
| **Exploratory** (development data) | Gradients decorrelate after one ε/4 step. bf16 vision rotates the gradient but not the attack outcome. A slight cross-context asymmetry. | [Checkpoint 2.3 report](docs/checkpoint-2.3-exploratory/REPORT.md) |
| **Exploratory negative** (16 development scenes) | The sign-consensus masks tested (sign votes over 1–8 random points at ±ε/4, 4 sparse on-grid points, 2 points at ±ε/16, and an fp32 vote) do **not** reproduce PGD's selection: the strongest recovers 5% of the selection gain [−5%, 14%]. Smoothed-gradient magnitude is worse than random. Matching PGD's per-patch density (16 px) recovers about a fifth of the gain. Other samplers and consensus rules were not tested. | [Checkpoint 3 pilot report](docs/checkpoint-3-mechanism/PILOT_REPORT.md) |
| **Not established** | The mechanism; generality to other models, budgets and precisions. | Report §7 |

## Exploratory follow-up after the confirmation: bf16 vs full fp32 (not a confirmation)

The confirmed study ran Qwen3-VL-2B in bf16. A later **exploratory** check asked whether bf16 evaluation could explain the result ([Checkpoint 5 report](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/ef2b6375ad36d5b4e9f3a9e32422c48f57571f6f/docs/checkpoint-5-qwen-fp32/REPORT.md), on the separate research branch `research/qwen-fp32-gpu`).

- **Design:**
  - **48 development scenes**, not new held-out data;
  - the unchanged 2.4 scene computation, in bf16 and in full fp32, on one CUDA GPU (NVIDIA A40, rented);
  - the checkpoint weights are stored in bf16, so the fp32 run is an exact up-cast with fp32 arithmetic;
  - TF32 off, no autocast;
  - protocol, gates and analysis committed before the run.
- **Result (pooled over both contexts, paired fp32 − bf16, 95% CI):**
  - **Selection advantage:** −0.19 logits [−0.61, 0.22]; 90% CI [−0.54, 0.16], inside the pre-registered ±1.0 equivalence margin. It is about +6 logits in each precision (bf16 6.10, fp32 5.91).
  - **Clean-gradient saliency inversion** (top-\|∇\| − random): −0.15 [−0.41, 0.09]; 90% CI [−0.37, 0.05], inside the ±0.5 margin. It is about −1.6 logits in each precision (bf16 −1.57, fp32 −1.73).
  - Both effects persist in fp32 in both contexts, and every paired difference is equivalent within its margin.
- **Caveats:**
  - The gate requiring the GPU bf16 run to reproduce the Mac (MPS) bf16 clean margins passed **exactly at its tolerances** (median 0.25, 95th percentile 0.75 logits).
  - One model, one GPU, development data.
  - This is **not** an independent confirmation and **not** a general precision-independence result. The Checkpoint 2.4 held-out confirmation above remains the primary result. The manuscript and frozen protocol are unchanged.

## Read next

- **Manuscript (preprint-style):** [`docs/paper/MANUSCRIPT.md`](docs/paper/MANUSCRIPT.md)
- **Full confirmation report:** [`docs/checkpoint-2.4-confirm/REPORT.md`](docs/checkpoint-2.4-confirm/REPORT.md)
- **Frozen protocol and hashes:** [`PROTOCOL.md`](docs/checkpoint-2.4-confirm/PROTOCOL.md), [`FROZEN.json`](docs/checkpoint-2.4-confirm/FROZEN.json)
- **One-page research summary:** [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md)

## Reproduce

**Without the model (Python 3.12, numpy and Pillow).** Every reported table and Figure 1 can be regenerated from the committed records; no tensors, weights or GPU are needed. One command checks all of it and writes nothing into the repository:

```bash
python tools/verify_reproduction.py
```

It verifies that:
- the 15 frozen Checkpoint 2.4 files match `FROZEN.json`, and none changed after the freeze commit;
- the held-out split regenerates exactly. On Apple silicon macOS, the platform it was built on, every file must be byte-identical. On other platforms PNG compression bytes depend on the zlib build, so each PNG must decode to identical RGB uint8 pixels, size and metadata, and every other file (manifest, metadata, pixel hashes) must still be byte-identical; see [`docs/PUBLICATION_AUDIT.md`](docs/PUBLICATION_AUDIT.md) §6;
- the frozen confirmation analysis, the per-group table, Figure 1 and the Checkpoint 3 pilot analysis regenerate from committed records. The table, figure and pilot analysis must be byte-identical; the confirmation JSON may differ only in its `input` field, which records an absolute path.

Continuous integration ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs this check and the model-free tests on every pull request. It never downloads model weights.

**Recorded environment** ([`requirements-ci.txt`](requirements-ci.txt), from `docs/models/measured-environment.txt`): Python 3.12.14, torch 2.15.0, torchvision 0.29.2, transformers 5.22.1, numpy 2.5.4, **Pillow 13.0.0**, accelerate 1.15.0, huggingface_hub 2.4.0, pytest 9.2.0.

```bash
python -m pip install -r requirements-ci.txt
```

All test suites run in one call, with no `PYTHONPATH` to set and no model:

```bash
python -m pytest -q
```

**Pillow versions.** The held-out split was generated with Pillow 13.0.0, as its `metadata.json` records, and `src/scene_confirm/pyproject.toml` now pins that version. The four development generators (`scene_gen`, `scene_v2`, `scene_simple`, `scene_v4`) still pin `Pillow==11.3.0`, the version their committed `checksums.json` files record. That historical difference is left as it is. The test suite regenerates all four development sets under Pillow 13.0.0 and compares them with the committed images pixel for pixel.

**With the model.** You need Apple silicon with about 12 GiB free and the pinned Qwen3-VL-2B weights (revision `89644892`) already in a local Hugging Face cache. The scripts never download them.
- By default, model-loading entry points refuse any `HF_HOME` other than the path the study used (`REQUIRED_HF_HOME` in `src/qwen_rgb/qwen_rgb/pins.py`). That is the configuration of every recorded run, and those files are unchanged.
- On another machine, opt in to your own cache by name and start the entry point through the `study_cache` launcher. The model id and revision stay pinned; only the cache directory changes, and the run's `environment` record shows the directory that was read:

```bash
VLM_STUDY_MODEL_CACHE=/absolute/path/to/hf-cache python -m study_cache path_probe confirm
```

- Setting `HF_HOME` alone is still refused. Without `VLM_STUDY_MODEL_CACHE`, `python -m study_cache <module>` behaves exactly like `python -m <module>`.
- The override has unit tests ([`tests/study_cache/`](tests/study_cache/)) but has not been exercised with a full model run on a second machine.

For commands below that import the packages directly, put them on the path:

```bash
export PYTHONPATH=$(ls -d src/*/ | tr '\n' ':')
```

The individual steps that `verify_reproduction.py` runs, if you want the outputs. Regenerate the held-out split:

```bash
python -m scene_confirm --output /tmp/confirm_check
```

Re-run the frozen analysis on the committed records, and redraw Figure 1 (these two overwrite the committed outputs in place; the JSON's `input` field will then hold your path):

```bash
python docs/checkpoint-2.4-confirm/analyze_confirm.py
```

```bash
python docs/checkpoint-2.4-confirm/make_figure.py
```

Re-run the Checkpoint 3 pilot analysis (exploratory) from its committed records:

```bash
python docs/checkpoint-3-mechanism/analyze_pilot.py
```

The full confirmation run takes about 2.7 h and needs the model. It refuses to start unless every frozen hash matches, and it refuses to overwrite earlier records:

```bash
python -m path_probe confirm
```

See [report §8](docs/checkpoint-2.4-confirm/REPORT.md#8-reproducibility) for the development-scene bit-exact smoke test and determinism notes.

## How the project got here

The project began as **Study 2**: does gradient sign alignment between two non-revealing text contexts predict whether an image attack transfers between them? That hypothesis, H\*, was pre-registered and stopped at Checkpoint 2.1 because too few transfer failures occurred to estimate it (4 against a floor of 40). It is reported as it stands ([`docs/checkpoint-2.1/`](docs/checkpoint-2.1/)).

Diagnosing why gradient geometry predicted so little led to the exploratory Checkpoint 2.3, which observed that single-step attacks and first-order reasoning fail on this model. A pre-registered random-mask control (2.3b) isolated coordinate selection. Checkpoint 2.4 then confirmed it on a newly generated, independently seeded split.

Every checkpoint keeps its plan, frozen records and deviations. Earlier results are never edited, only annotated with dated notes.

| Path | Content |
|---|---|
| `docs/checkpoint-2.4-confirm/` | **Held-out confirmation:** protocol, freeze, records, analysis, report, figure |
| `docs/checkpoint-2.3-exploratory/` | Exploratory path, precision and mask-control studies |
| `docs/checkpoint-3-mechanism/` | Mechanism plan, development-only pilot and its report (negative) |
| `docs/checkpoint-0` … `2.2-draft` | Study 2 design, measurement and attack protocols, H\*, audits |
| `docs/research/` | Literature review and prior-art matrix |
| `src/` | Scene generators, scoring, attack and probe code |
| `fixtures/` | Rendered development and confirmation scenes, with manifests and hashes |

## License

| Material | License |
|---|---|
| Original source code and tests: `src/`, `tests/`, `tools/`, `conftest.py`, the `.py` scripts under `docs/`, CI configuration | [MIT](LICENSE) |
| Original synthetic scenes and their manifests (`fixtures/`), experimental records, figures, and research documentation under `docs/` | [CC BY 4.0](LICENSE-CC-BY-4.0.txt) |
| The draft manuscript, [`docs/paper/MANUSCRIPT.md`](docs/paper/MANUSCRIPT.md) | **Not licensed for reuse yet** (all rights reserved), pending a publication decision. You may read and cite it. |

**Not covered.** Nothing here relicenses third-party material:
- **Not in the repository:** the Qwen3-VL-2B-Instruct and SmolVLM2 weights, which are downloaded from their publishers and remain under their own licenses, and the Python dependencies in `requirements-ci.txt`.
- **Model outputs:** some records (for example the `predictions.jsonl` files) contain text and scores produced by those models. They are covered by CC BY 4.0 only to the extent we hold rights in them.
- **Quotations:** short quotations of published work in `docs/research/` remain their authors'.
- **CLEVR:** discussed in `docs/data/` but never downloaded or included.

## Citation

Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

```
Cavaletto, F. (2026). Which Coordinates, Not How Many: Matched-Count Evidence on the PGD–FGSM Gap in a Vision–Language Model.
https://github.com/fcavaletto/vlm-crossmodal-attack-transfer
```
