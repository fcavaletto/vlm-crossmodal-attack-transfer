"""Check the EMP-1 smoke run against MEMORY_ASSESSMENT.md section 4. Reads only.

Frozen inputs (read, never written): docs/checkpoint-2.1/development/records.jsonl
(clean and attacked official scores) and docs/checkpoint-1.8/qwen_rgb_loss/progress.json.

Run: python3 docs/checkpoint-2.3-exploratory/check_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "docs/checkpoint-2.3-exploratory/paths_smoke"
GIB = 1024**3
CTX = ("neutral", "helpful_nonrevealing")
FROZEN_KEY = {"fgsm": ("fgsm", 1), "random": ("random", 1), 3: ("pgd", 3)}


def main() -> None:
    prog = json.loads((SMOKE / "progress.json").read_text())
    rows = [json.loads(x) for x in (SMOKE / "records.jsonl").read_text().splitlines()] if (SMOKE / "records.jsonl").exists() else []
    frozen = {}
    for line in (ROOT / "docs/checkpoint-2.1/development/records.jsonl").read_text().splitlines():
        r = json.loads(line)
        frozen[(r["scene_id"], r["method"], r["steps"])] = r
    out: dict = {"status": prog.get("status"), "stopped_reason": prog.get("stopped_reason"), "error": prog.get("error")}

    mon = prog.get("memory_monitor") or {}
    base = prog.get("footprint_before_load") or {}
    tl = mon.get("timeline") or []
    out["memory"] = {
        "process_peak_footprint_GiB": round((mon.get("process_peak_footprint") or 0) / GIB, 3),
        "footprint_end_lifetime_max_GiB": round(((prog.get("footprint_end") or {}).get("lifetime_max") or 0) / GIB, 3),
        "footprint_before_load_GiB": round((base.get("current") or 0) / GIB, 3),
        "min_available_GiB": round((mon.get("min_available") or 0) / GIB, 3),
        "pressure_samples_by_level": mon.get("by_level"),
        "max_pressure_level": max((int(k) for k in (mon.get("by_level") or {}) if k != "None"), default=None),
        "seconds_at_warn_or_worse": round(sum(v for k, v in (mon.get("by_level") or {}).items() if k not in ("1", "None")) * 0.5, 1),
        "swap_growth_GiB": round((mon.get("max_swap_growth") or 0) / GIB, 3),
        "samples": mon.get("samples"),
        "abort_reason": mon.get("abort_reason"),
        "emp1_preload_available_GiB": round(((prog.get("emp1_preload") or {}).get("sample") or {}).get("available", 0) / GIB, 3),
        "frozen_guard_bf16_permitted": prog.get("frozen_guard_bf16_permitted"),
    }

    # Reproduction: clean and attacked official scores, mode B, neutral source.
    rep = {"clean": [], "attacks": []}
    for r in rows:
        if r.get("mode") != "B":
            continue
        sid = r["scene_id"]
        if r["kind"] == "clean":
            f = frozen[(sid, "pgd", 3)]["clean"]
            for c in CTX:
                m = r["scores"][c]["official"]
                rep["clean"].append({"context": c, "nll_equal": m["nll"] == f[c]["nll"], "margin_equal": m["margin"] == f[c]["margin"],
                                     "decision_equal": m["chosen"] == f[c]["decision"], "nll": [m["nll"], f[c]["nll"]]})
        if r.get("source") != "neutral":
            continue
        key = r["kind"] if r["kind"] in ("fgsm", "random") else (r.get("k") if r["kind"] == "pgd" else None)
        if key not in FROZEN_KEY or "official" not in r["scores"]["neutral"]:
            continue
        f = frozen[(sid, *FROZEN_KEY[key])]["perturbed"]
        for c in CTX:
            m = r["scores"][c]["official"]
            rep["attacks"].append({"attack": str(key), "context": c, "nll_equal": m["nll"] == f[c]["nll"],
                                   "margin_equal": m["margin"] == f[c]["margin"], "decision_equal": m["chosen"] == f[c]["decision"],
                                   "nll": [m["nll"], f[c]["nll"]]})
    allrows = rep["clean"] + rep["attacks"]
    rep["n_compared"] = len(allrows)
    rep["n_bit_exact"] = sum(x["nll_equal"] and x["margin_equal"] and x["decision_equal"] for x in allrows)
    clean_rows = [r for r in rows if r["kind"] == "clean" and r.get("mode") == "B"]
    rep["determinism_repeat_max_abs_diff"] = max(
        (max(v.values()) for r in clean_rows for v in r["repeat_float"].values()), default=None)
    rep["prog_reproduction_check"] = prog.get("reproduction_check")
    out["reproduction"] = rep

    st3 = (prog.get("stages") or {}).get("V32") or {}
    v_clean = [r for r in rows if r.get("mode") == "V32" and r["kind"] == "clean"]
    out["fp32_vision"] = {
        "skipped": st3.get("skipped"), "measured": st3.get("measured"), "pressure_at_check": st3.get("pressure_level_at_check"),
        "available_at_check_GiB": round(st3.get("available_bytes", 0) / GIB, 3) if st3.get("available_bytes", 0) > 0 else st3.get("available_bytes"),
        "required_GiB": round(st3.get("required_bytes", 0) / GIB, 3), "converted": st3.get("converted_elements"),
        "restored": st3.get("restored_elements"), "dtype_after": st3.get("visual_dtype_after"),
        "dtype_restored": st3.get("visual_dtype_restored"), "restore_check": st3.get("restore_check"),
        "grad_agreement_with_B": v_clean[0].get("agreement_with_B") if v_clean else None,
        "n_B_cross_rows": sum(1 for r in rows if r.get("mode") == "B_cross"),
        "seconds": st3.get("seconds"),
    }
    sc = prog.get("scenes") or {}
    out["runtime"] = {"load_seconds": (prog.get("loaded_model") or {}).get("load_seconds"),
                      "elapsed_seconds_after_load": prog.get("elapsed_seconds"),
                      "scene_seconds_B": {k: v.get("seconds") for k, v in sc.items()},
                      "scene_seconds_V32": {k: v.get("seconds_V32") for k, v in sc.items()},
                      "pgd_gradient_seconds": [r.get("seconds") for r in rows if r["kind"] == "pgd_gradient"]}
    out["limitations"] = prog.get("limitations")
    (SMOKE / "smoke_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
