# Checkpoint 2.0 CUDA planning estimate

DEVELOPMENT ONLY. This file is a planning estimate. It is not a CUDA measurement. No calibration attack was run. No held-out image was opened. The loss, the decision, the pixel path, and the attack step were not changed.

## Quoted cost

The class is one NVIDIA L4 on an AWS EC2 `g6.2xlarge` in `us-east-1`, Linux on-demand, at **$0.9776 per hour**. The instance has 8 vCPUs and 32 GiB of host memory. The L4 is the 24 GB class. Listings also describe that card as about 22 GiB; the memory section uses 22 GiB as the smaller published floor. The rate excludes EBS, a public IPv4 address, data transfer, and tax.

Holori lists this rate as updated 2026-11-06. Vantage and Cloud Credits list the same $0.9776 figure.

| Study | Scenes | Estimated wall time | On-demand cost |
| --- | ---: | ---: | ---: |
| Development, Stage B grid | 96 | 47,272.982 s (13.131 h) | $12.84 |
| Held-out planning count | 256 | 126,055.242 s (35.015 h) | $34.23 |

Both rows use `assumed_cuda_seconds_per_mps_second = 1`. The machine-readable copy is `docs/checkpoint-2.0/cuda/planning_estimate.json`.

## What was scaled

Checkpoint 1.9 recorded MPS, bfloat16, eager attention. The sixteen Stage B scene records sum to 7,878.226 seconds, which is 492.389125 seconds per scene. The eight Stage A records sum to 326.408 seconds. Stage A is a single-step grid, so it is not the scaler. The finishing process recorded 7,447.295 seconds of wall clock and 3.626 seconds of load. That wall clock leaves out an earlier process, so it is not the scaler. The 3.626 seconds are added once, as an MPS load time.

The task being priced is that Stage B grid: source context `neutral`, evaluation under `neutral` and `helpful_nonrevealing`, budgets `1/255`, `2/255`, and `4/255`, seven projected steps of length epsilon/4 from the clean image, no restarts, plus one positive sign step and one matched random-sign step. Scenes are sequential. The 96-scene row applies that grid to every `development_v4` scene. It is the full three-budget grid, which is wider than the two-budget calibration sample.

The held-out count is 256, the sealed-set size written in `docs/research/RESEARCH_PROPOSAL.md` section 12 ("384 scene seeds: 128 development, 256 sealed held-out"). The development set that exists is 96 scenes, so the 128 in that sentence is not the count used here. No sealed directory was listed or opened. A different sealed count uses the same 492.389125 seconds per scene.

## Sensitivities

These ratios are assumptions. A ratio of 0.5 means the estimate uses half the measured MPS scene time. A ratio of 2 means it uses twice that time. The load stays 3.626 seconds in every row. The GPU class and the $0.9776 rate are the same.

| Study | Ratio | Wall seconds | Cost |
| --- | ---: | ---: | ---: |
| 96 scenes | 0.5 | 23,638.304 | $6.42 |
| 96 scenes | 2 | 94,542.338 | $25.67 |
| 256 scenes | 0.5 | 63,029.434 | $17.12 |
| 256 scenes | 2 | 252,106.858 | $68.46 |

## GPU memory

This is an estimate. This machine has no CUDA device, and no CUDA allocator high-water mark was read.

Bfloat16 weights are 4,255,064,064 bytes. The Checkpoint 1.9 MPS snapshot peaked at 4,072.9 MiB current and 4,749.7 MiB driver-allocated. Those snapshots were taken on MPS, after the runner released its cache between steps. They are not a backward peak and they are not CUDA.

The unchanged Checkpoint 1.5 lines, recomputed from `docs/checkpoint-1.5/run_model_feasibility.py`, are:

| Line | Bytes |
| --- | ---: |
| OS floor | 6,442,450,944 |
| Spare above bfloat16 weights | 9,623,773,184 |
| Bfloat16 calibrated projection | 11,509,447,200 |
| Bfloat16 staging double copy | 8,510,128,128 |
| Float32 calibrated projection | 20,871,410,752 |

The guard was not lowered. The calibrated projection is a unified-memory OS-available line, carried forward from the Checkpoint 1 ratio. It is included so a later reading can see that 11,509,447,200 bytes still sits under a 22 GiB floor (23,622,320,128 bytes), as do the weight bytes and the observed driver snapshot of 4,980,421,427 bytes. Fitting under that floor is arithmetic on published and recorded numbers. It is not evidence that an L4 backward succeeded.

`g6.2xlarge` is the quoted class because its host has 32 GiB. The unchanged bfloat16 line asks for 11,509,447,200 bytes still available to the OS before a load. A 16 GiB `g6.xlarge`, at $0.8048 per hour on the same listing sites, can miss that line, so it is not the quoted class. Neither instance was started. The existing budget decision still requires MPS, and `attack_feasibility` still requests MPS. Those two facts were left in place.
