"""Main manuscript figure (presentation only; it is not part of any frozen analysis).

    python docs/paper/figures/make_main_figure.py

Panel A is a schematic with illustrative values: how 8-bit rounding turns PGD-3 into FGSM's perturbation
restricted to the coordinates where all three step signs agree. The schematic shows the case in which every
step has a nonzero gradient; zero-gradient steps can produce rounding ties, documented in the manuscript (§3).
Panel B plots the two held-out matched contrasts from the frozen Checkpoint 2.4 outputs:
- docs/checkpoint-2.4-confirm/analysis/confirm_analysis.json (means and 95% CIs, pooled over source contexts);
- docs/checkpoint-2.4-confirm/analysis/confirm_per_group.csv (one point per scene group).
The script checks that the per-group points reproduce the frozen means before drawing.
Standard library only; writes figures/figure1_main.svg.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIRM = ROOT / "docs" / "checkpoint-2.4-confirm" / "analysis"
OUT = HERE / "figure1_main.svg"
CTX = ("neutral", "helpful_nonrevealing")

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # selection, saliency (reference palette slots 1 and 2)
KEEP_FILL = "#e3edfa"  # light tint of BLUE for kept coordinates
FONT = "Helvetica, Arial, sans-serif"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, fill=INK, anchor="start", weight="normal", style="normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-style="{style}" stroke="{SURFACE}" stroke-width="3" paint-order="stroke" '
            f'stroke-linejoin="round">{esc(s)}</text>')


def load() -> tuple[dict, list[dict]]:
    a = json.loads((CONFIRM / "confirm_analysis.json").read_text())
    with (CONFIRM / "confirm_per_group.csv").open() as fh:
        g = list(csv.DictReader(fh))
    return a, g


def contrasts(a: dict, g: list[dict]) -> list[dict]:
    pooled = a["margin_drop_source"]["pooled"]
    spec = [("selection_pgd_minus_rand", "pgd", "PGD-3 subset − random subset", "selection", BLUE),
            ("magnitude_topk_minus_rand", "topk", "Top-|∇| subset − random subset", "clean-gradient magnitude", ORANGE)]
    out = []
    for key, m, label, sub, color in spec:
        pts = [sum(float(r[f"D_{m}_{c}"]) - float(r[f"D_rand_{c}"]) for c in CTX) / len(CTX) for r in g]
        est = pooled[key]["mean"]
        if len(pts) != est["n"] or abs(sum(pts) / len(pts) - est["estimate"]) > 1e-4:
            raise SystemExit(f"per-group points do not reproduce the frozen mean for {key}")
        out.append({"label": label, "sub": sub, "color": color, "pts": pts,
                    "est": est["estimate"], "ci": est["ci95"], "n": est["n"]})
    return out


def panel_a(x0: float, y0: float) -> list[str]:
    s = [text(x0, y0, "A", 18, weight="bold"),
         text(x0 + 22, y0, "8-bit rounding turns PGD-3 into a subset of FGSM", 15, weight="bold"),
         text(x0 + 22, y0 + 19, "Schematic: six RGB-channel coordinates, illustrative values, all steps nonzero", 12, INK2, style="italic")]

    # One pixel = three RGB-channel coordinates.
    py = y0 + 46
    s.append(text(x0, py + 13, "One pixel has three coordinates:", 12, INK2))
    for i, (ch, col) in enumerate((("R", "#d9534f"), ("G", "#3a9d5d"), ("B", "#3b6fc4"))):
        bx = x0 + 196 + i * 30
        s.append(f'<rect x="{bx}" y="{py}" width="24" height="18" rx="3" fill="{SURFACE}" stroke="{col}" stroke-width="1.6"/>')
        s.append(text(bx + 12, py + 13.5, ch, 11, col, "middle", "bold"))
    s.append(text(x0 + 290, py + 13, "3 × 384 × 512 = 589,824 per image", 12, INK2))

    cols = ["R", "G", "B", "R", "G", "B"]
    fgsm = [+1, -1, +1, -1, +1, -1]          # sign of the clean gradient = FGSM step sign
    step2 = [+1, +1, -1, -1, +1, +1]
    step3 = [+1, -1, +1, -1, -1, +1]
    disp = [fgsm[j] + step2[j] + step3[j] for j in range(6)]  # in units of ε/4
    rounded = [(1 if d == 3 else -1 if d == -3 else 0) for d in disp]
    keep = [r != 0 for r in rounded]

    def sgn(v):
        return "+" if v > 0 else "−"

    def quarter(v):
        return {3: "+¾", 1: "+¼", -1: "−¼", -3: "−¾"}[v]

    def level(v):
        return "+1" if v > 0 else "−1" if v < 0 else "0"

    rows = [("Step 1 sign (= FGSM sign)", [sgn(v) for v in fgsm]),
            ("Step 2 sign", [sgn(v) for v in step2]),
            ("Step 3 sign", [sgn(v) for v in step3]),
            ("PGD-3 displacement (ε)", [quarter(v) for v in disp]),
            ("PGD-3 after rounding (levels)", [level(v) for v in rounded]),
            ("FGSM after rounding (levels)", [level(v) for v in fgsm])]

    gx, cw, rh = x0 + 222, 48, 27
    gy = py + 62
    for j, c in enumerate(cols):
        cx = gx + j * cw
        if keep[j]:
            s.append(f'<rect x="{cx + 3}" y="{gy - 4}" width="{cw - 6}" height="{rh * len(rows) + 30}" rx="5" '
                     f'fill="{KEEP_FILL}" stroke="{BLUE}" stroke-width="1.4"/>')
        s.append(text(cx + cw / 2, gy + 14, c, 12, INK2, "middle", "bold"))
    for k, lab in enumerate(("pixel 1", "pixel 2")):
        mx = gx + (3 * k + 1.5) * cw
        s.append(f'<line x1="{gx + 3 * k * cw + 6}" y1="{gy - 14}" x2="{gx + (3 * k + 3) * cw - 6}" y2="{gy - 14}" stroke="{MUTED}" stroke-width="1"/>')
        s.append(text(mx, gy - 19, lab, 11, INK2, "middle"))
    for i, (name, vals) in enumerate(rows):
        ry = gy + 30 + i * rh
        if i == 3:
            s.append(f'<line x1="{x0}" y1="{ry - 8}" x2="{gx + 6 * cw}" y2="{ry - 8}" stroke="{GRID}" stroke-width="1"/>')
        s.append(text(x0, ry + 9, name, 12, INK))
        for j, v in enumerate(vals):
            emph = keep[j] and i >= 4 and v != "0"
            s.append(text(gx + j * cw + cw / 2, ry + 9, v, 13, BLUE if emph else INK, "middle",
                          "bold" if emph else "normal"))

    ny = gy + 30 + len(rows) * rh + 22
    notes = [
        "Case shown: every step's gradient is nonzero and values stay inside [0, 1]. Then steps of ε/4",
        "never reach the ε-ball boundary, so each displacement is ±¼ε or ±¾ε, and rounding to the 8-bit",
        "grid (spacing ε) sends ±¼ε to 0 and ±¾ε to ±ε. A coordinate survives only if all three step",
        "signs agree, and it then carries FGSM's sign (shaded). A zero-gradient step can leave ±½ε,",
        "a rounding tie (observed: 0–7 coordinates per image; §3). The compared masks (PGD-3, random,",
        "top-|∇|) each keep k of FGSM's coordinates at one level with FGSM's sign there; only the choice",
        "of coordinates differs, so counts of + and − can differ between masks.",
    ]
    for i, n in enumerate(notes):
        s.append(text(x0, ny + i * 17, n, 12, INK2))
    return s


def panel_b(x0: float, y0: float, w: float, cs: list[dict]) -> list[str]:
    s = [text(x0, y0, "B", 18, weight="bold"),
         text(x0 + 22, y0, "Held-out matched contrasts (Qwen3-VL-2B)", 15, weight="bold"),
         text(x0 + 22, y0 + 19, "48 independent scene groups; difference in margin drop, logits", 12, INK2, style="italic")]
    lo, hi = -5.0, 11.0
    ax0, ax1 = x0 + 10, x0 + w - 10

    def X(v):
        return ax0 + (ax1 - ax0) * (v - lo) / (hi - lo)

    top, band = y0 + 64, 128
    bottom = top + band * len(cs)
    for t in range(-4, 11, 2):
        s.append(f'<line x1="{X(t):.1f}" y1="{top - 6}" x2="{X(t):.1f}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>')
        s.append(text(X(t), bottom + 17, f"{t:+d}" if t else "0", 11, INK2, "middle"))
    s.append(f'<line x1="{X(0):.1f}" y1="{top - 6}" x2="{X(0):.1f}" y2="{bottom}" stroke="{INK2}" stroke-width="1.2"/>')
    s.append(text((ax0 + ax1) / 2, bottom + 36, "Margin-drop difference (logits); right = more damage than random", 12, INK, "middle"))

    for i, c in enumerate(cs):
        by = top + i * band
        s.append(text(ax0, by + 12, c["label"], 13, INK, weight="bold"))
        s.append(text(ax0, by + 28, c["sub"], 12, INK2))
        cy = by + 74
        for j, v in enumerate(sorted(c["pts"])):  # deterministic vertical spread
            dy = ((j * 7) % 11 - 5) * 2.6
            s.append(f'<circle cx="{X(v):.1f}" cy="{cy + dy:.1f}" r="3.4" fill="{c["color"]}" fill-opacity="0.28" '
                     f'stroke="{c["color"]}" stroke-opacity="0.75" stroke-width="0.8"/>')
        lo_ci, hi_ci = c["ci"]
        ey = cy + 30
        s.append(f'<line x1="{X(lo_ci):.1f}" y1="{ey}" x2="{X(hi_ci):.1f}" y2="{ey}" stroke="{INK}" stroke-width="2.2"/>')
        for e in (lo_ci, hi_ci):
            s.append(f'<line x1="{X(e):.1f}" y1="{ey - 5}" x2="{X(e):.1f}" y2="{ey + 5}" stroke="{INK}" stroke-width="2"/>')
        s.append(f'<circle cx="{X(c["est"]):.1f}" cy="{ey}" r="5.5" fill="{c["color"]}" stroke="{SURFACE}" stroke-width="1.5"/>')
        lab = f'mean {c["est"]:+.2f}  [{lo_ci:.2f}, {hi_ci:.2f}]'.replace("-", "−")
        if c["est"] > 0:
            s.append(text(X(lo_ci) - 10, ey + 4.5, lab, 12, INK, "end", "bold"))
        else:
            s.append(text(X(hi_ci) + 10, ey + 4.5, lab, 12, INK, "start", "bold"))

    ny = bottom + 62
    notes = ["Dots: one per scene group, averaged over the two source contexts; random = mean of 8 seeds.",
             "Bar: mean and 95% percentile bootstrap CI over groups (9,999 draws), from the frozen analysis.",
             "Within each source context, PGD-3's subset beat all 8 random subsets in 48 of 48 groups."]
    for i, n in enumerate(notes):
        s.append(text(x0 + 10, ny + i * 17, n, 12, INK2))
    return s


def describe(cs: list[dict]) -> str:
    """Accessibility description; build_pdf.py reuses it as the image's alt text."""
    def ci(c):
        return f'mean {c["est"]:+.2f}, 95% interval {c["ci"][0]:+.2f} to {c["ci"][1]:+.2f}'
    return ("Panel A: schematic with illustrative values for six RGB-channel coordinates in the case where all three "
            "PGD steps have nonzero gradient. In that case, after 8-bit rounding, PGD-3 keeps FGSM's sign on the "
            "coordinates where all three step signs agree and is zero elsewhere. Steps with zero gradient can instead "
            "produce rounding ties; the text documents 0 to 7 such coordinates per image. Panel B: held-out contrasts "
            "in margin drop for 48 scene groups, one dot per group. PGD-3 subset minus random subset: "
            f"{ci(cs[0])}. Top-gradient-magnitude subset minus random subset: {ci(cs[1])}.")


def build() -> str:
    a, g = load()
    cs = contrasts(a, g)
    W, H = 760, 940
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}" '
           f'role="img" aria-labelledby="t d">',
           '<title id="t">PGD-3 as a subset of FGSM, and the held-out matched contrasts</title>',
           f'<desc id="d">{describe(cs)}</desc>',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>']
    out += panel_a(24, 34)
    out.append(f'<line x1="24" y1="473" x2="{W - 24}" y2="473" stroke="{GRID}" stroke-width="1"/>')
    out += panel_b(24, 511, W - 48, cs)
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    OUT.write_text(build())
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
