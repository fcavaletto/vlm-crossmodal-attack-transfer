"""Does upcasting bf16 logits to float32 remove the Checkpoint 2.1 ties? No.

Three parts. None loads Qwen or runs an attack.

1. Records. Every recorded margin in docs/checkpoint-2.1/development/records.jsonl
   is checked against the bf16 grid. The scorer already upcasts the logit row to
   float32 before log_softmax (src/dev_pilot/dev_pilot/scoring.py, line 25), so
   the ties happened after that upcast.
2. Synthetic arithmetic (numpy, no weights). One final projection, logits = W h,
   is computed three ways from the same bf16 inputs: rounded to bf16 then upcast
   (the current path), accumulated and kept in float32 ("fp32 readout"), and in
   float64 from the unrounded inputs (reference). It shows which step makes the
   grid and what an fp32 readout can and cannot recover. The magnitudes are
   illustrative, not Qwen's.
3. Endpoint resolution. For the proposed continuous endpoint (labeled-color NLL),
   the bf16 grid's rounding error is bounded from the recorded NLLs and compared
   with the spread of the development outcome. This uses development outcomes
   only for their spread, never their relation to any predictor.

Run: python3 docs/audit-2026-11-08/precision_check.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECORDS = ROOT / "docs/checkpoint-2.1/development/records.jsonl"


def bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (nearest, ties to even) and return it as float32."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def finest_grid(value: float) -> int | None:
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        if abs(value * k - round(value * k)) < 1e-5 * max(1.0, abs(value * k)):
            return k
    return None


def part_records() -> dict:
    rows = [json.loads(line) for line in RECORDS.read_text().splitlines()]
    margins, ties = [], []
    for row in rows:
        for part in ("clean", "perturbed"):
            for context, view in row[part].items():
                if view.get("margin") is None:
                    continue
                margins.append(view["margin"])
                if abs(view["margin"]) < 1e-9:
                    ties.append({
                        "scene_id": row["scene_id"], "method": row["method"], "steps": row["steps"],
                        "part": part, "context": context, "label": row["answer"],
                        "decision": view["decision"], "label_won_tie_break": view["decision"] == row["answer"],
                    })
    grid = Counter(finest_grid(m) for m in margins)
    return {
        "margins": len(margins),
        "finest_grid_1_over_k": {str(k): grid[k] for k in sorted(k for k in grid if k is not None)},
        "off_every_grid_to_1_over_128": grid[None],
        "exact_ties": len(ties),
        "ties_label_won": sum(t["label_won_tie_break"] for t in ties),
        "ties_label_lost": sum(not t["label_won_tie_break"] for t in ties),
        "ties": ties,
    }


def part_synthetic(trials: int = 20000, width: int = 2048, seed: int = 20261108) -> dict:
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((trials, width)).astype(np.float32)
    w0 = (rng.standard_normal((trials, width)) * 0.02).astype(np.float32)
    # The second row is the first plus a change that gives true margins with SD near 1 nat.
    w1 = (w0 + rng.standard_normal((trials, width)).astype(np.float32) * 0.022).astype(np.float32)
    shift = np.float32(40.0)  # puts the logits in the bf16 binade [32, 64), spacing 0.25

    def logits(wa, ha, dtype):
        return (np.einsum("ij,ij->i", wa.astype(dtype), ha.astype(dtype)) + dtype(shift))

    reference = logits(w0, h, np.float64) - logits(w1, h, np.float64)
    hb, w0b, w1b = bf16(h), bf16(w0), bf16(w1)
    readout0 = logits(w0b, hb, np.float32)
    readout1 = logits(w1b, hb, np.float32)
    fp32_readout = (readout0 - readout1).astype(np.float64)
    current = (bf16(readout0).astype(np.float64) - bf16(readout1).astype(np.float64))
    upcast_after = current.astype(np.float32).astype(np.float64)  # what .float() adds: nothing
    return {
        "trials": trials,
        "true_margin_sd": float(reference.std()),
        "current_path_exact_ties": int((current == 0).sum()),
        "current_path_on_quarter_grid": int(np.all(np.abs(current * 4 - np.round(current * 4)) < 1e-9)),
        "upcast_after_rounding_identical_to_current": bool(np.array_equal(upcast_after, current)),
        "fp32_readout_exact_ties": int((fp32_readout == 0).sum()),
        "sign_flips_vs_reference": {
            "current_path": int((np.sign(current) != np.sign(reference)).sum()),
            "fp32_readout": int((np.sign(fp32_readout) != np.sign(reference)).sum()),
        },
        "abs_error_vs_reference_sd": {
            "current_path": float((current - reference).std()),
            "fp32_readout": float((fp32_readout - reference).std()),
        },
        "note": (
            "fp32_readout still carries the error from rounding W and h to bf16. In Qwen, h is "
            "also the output of 28 bf16 layers, whose accumulated error this toy does not model."
        ),
    }


def part_endpoint() -> dict:
    rows = [json.loads(line) for line in RECORDS.read_text().splitlines()]
    pgd3 = [r for r in rows if (r["method"], r["steps"]) == ("pgd", 3)]
    spacing = 0.25  # most margins are on the 1/4 grid; see part 1
    per_logit_sd = spacing / math.sqrt(12)
    bounds, deltas = [], []
    for row in pgd3:
        clean = row["clean"]["helpful_nonrevealing"]["nll"]
        pert = row["perturbed"]["helpful_nonrevealing"]["nll"]
        deltas.append(pert - clean)
        for nll in (clean, pert):
            p = math.exp(-nll)
            # d NLL / d logit_label = -(1 - p); sum over others of p_j^2 <= (1 - p)^2
            bounds.append(per_logit_sd * math.sqrt(2.0) * (1.0 - p))
    deltas = np.array(deltas)
    delta_noise_sd = math.sqrt(2.0) * float(np.sqrt(np.mean(np.square(bounds))))
    return {
        "scenes": len(pgd3),
        "delta_helpful_nll_sd": float(deltas.std(ddof=1)),
        "delta_helpful_nll_quantiles": {q: float(np.quantile(deltas, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "rounding_sd_upper_bound_per_delta": delta_noise_sd,
        "rounding_share_of_variance_upper_bound": delta_noise_sd**2 / float(deltas.var(ddof=1)),
    }


def main() -> None:
    result = {
        "schema": "audit.precision_check.v1",
        "records": part_records(),
        "synthetic": part_synthetic(),
        "endpoint_resolution": part_endpoint(),
    }
    out = HERE / "precision_check.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result["records"].items() if k != "ties"}, indent=2))
    print(json.dumps(result["synthetic"], indent=2))
    print(json.dumps(result["endpoint_resolution"], indent=2))


if __name__ == "__main__":
    main()
