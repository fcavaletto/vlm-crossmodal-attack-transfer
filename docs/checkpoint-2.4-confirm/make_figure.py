"""Central figure for the Checkpoint 2.4 report (presentation only; not part of the frozen analysis).

Run: python docs/checkpoint-2.4-confirm/make_figure.py [confirm_analysis.json] [per_group.csv] [out.svg]
Panel A: mean source-context margin drop (95% group-bootstrap CI) for dense FGSM and three matched-count
subsets of FGSM's support, development vs held-out confirmation.
Panel B: per group, PGD-3's subset vs the mean of eight random subsets with the same coordinate count.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CTX = ("neutral", "helpful_nonrevealing")
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"


def rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def pooled(r: dict, m: str) -> float:
    return sum(float(r[f"D_{m}_{c}"]) for c in CTX) / 2


def headline(a: dict) -> str:
    d = a.get("decisions", {})
    p1 = d.get("P1_selection_beats_matched_random", {}).get("supported")
    s2 = d.get("S2_topk_gradient_worse_than_random", {}).get("supported")
    if p1 and s2:
        return "Same coordinate count, different coordinates: PGD-3 picks damaging ones, gradient saliency picks weak ones"
    if p1:
        return "Same coordinate count, different coordinates: PGD-3 picks damaging ones"
    return "PGD-3's coordinates vs matched random and gradient-saliency coordinates"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(dev_a: dict, dev_g: list[dict], con_a: dict, con_g: list[dict], con_label: str) -> str:
    W, H = 1080, 520
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="Inter, Helvetica, Arial, sans-serif" role="img" aria-labelledby="t d">',
           '<title id="t">PGD-3 advantage comes from which coordinates it perturbs</title>',
           '<desc id="d">Panel A: mean source-context margin drop for FGSM and three matched-count subsets, development '
           'vs confirmation. Panel B: per-group PGD-3 subset damage against matched random subsets.</desc>',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="24" y="34" font-size="19" font-weight="600" fill="{INK}">{esc(headline(con_a))}</text>',
           f'<text x="24" y="58" font-size="13" fill="{INK2}">Qwen3-VL-2B (bf16), ε = 1/255, official 8-bit scoring. '
           'Source-context margin drop, pooled over the two source contexts. '
           'Unit = counterfactual group (n = 48 per split).</text>']

    # Legend
    lx, ly = 24, 84
    out += [f'<circle cx="{lx + 6}" cy="{ly}" r="5.5" fill="{ORANGE}" fill-opacity="0.12" stroke="{ORANGE}" stroke-width="2"/>',
            f'<text x="{lx + 18}" y="{ly + 4}" font-size="13" fill="{INK}">Development (exploratory, Checkpoint 2.3b)</text>',
            f'<circle cx="{lx + 330}" cy="{ly}" r="5.5" fill="{BLUE}" stroke="#ffffff" stroke-width="2"/>',
            f'<text x="{lx + 342}" y="{ly + 4}" font-size="13" fill="{INK}">{esc(con_label)}</text>']

    # Panel A
    ax0, ax1, ay0 = 190, 560, 140
    items = [("fgsm", "Dense FGSM", ("every coordinate with a", "nonzero clean gradient")),
             ("rand", "Random subset", ("k coords drawn uniformly from", "FGSM's support (mean of 8 seeds)")),
             ("topk", "Top-|∇| subset", ("the k coords of FGSM's support", "with largest |clean NLL gradient|")),
             ("pgd", "PGD-3 subset", ("the k coords PGD-3 keeps", "(the official PGD-3 image)"))]
    xmax = 12.0
    for blk in (dev_a, con_a):
        for m, *_ in items:
            hi = blk["margin_drop_source"]["pooled"][f"D_{m}"]["mean"]["ci95"][1]
            xmax = max(xmax, (hi or 0) + 1)
    xmax = float(int(xmax / 2 + 1) * 2)

    def X(v):
        return ax0 + (ax1 - ax0) * v / xmax

    out.append(f'<text x="24" y="{ay0 - 22}" font-size="14" font-weight="600" fill="{INK}">A · Mean margin drop (95% CI)</text>')
    band = 66
    for i in range(int(xmax / 2) + 1):
        v = 2 * i
        out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{ay0 - 6}" y2="{ay0 + band * 4 - 10}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{X(v):.1f}" y="{ay0 + band * 4 + 6}" font-size="11" fill="{MUTED}" text-anchor="middle">{v}</text>')
    out.append(f'<text x="{(ax0 + ax1) / 2:.0f}" y="{ay0 + band * 4 + 26}" font-size="12" fill="{INK2}" text-anchor="middle">'
               'clean margin − perturbed margin (logits)</text>')
    for j, (m, name, note) in enumerate(items):
        yc = ay0 + j * band + 16
        out.append(f'<text x="24" y="{yc - 2}" font-size="13" font-weight="600" fill="{INK}">{name}</text>')
        out.append(f'<text x="24" y="{yc + 14}" font-size="10.5" fill="{MUTED}">{esc(note[0])}</text>')
        out.append(f'<text x="24" y="{yc + 27}" font-size="10.5" fill="{MUTED}">{esc(note[1])}</text>')
        for blk, color, dy, filled, lab in ((dev_a, ORANGE, -8, False, "Development"), (con_a, BLUE, 10, True, con_label)):
            s = blk["margin_drop_source"]["pooled"][f"D_{m}"]["mean"]
            e, (lo, hi) = s["estimate"], s["ci95"]
            y = yc + dy
            tip = f"{lab} · {name}: {e:.2f} [{lo:.2f}, {hi:.2f}]"
            out.append(f'<g><title>{esc(tip)}</title>'
                       f'<line x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
                       + (f'<circle cx="{X(e):.1f}" cy="{y}" r="5.5" fill="{color}" stroke="#ffffff" stroke-width="2"/>' if filled else
                          f'<circle cx="{X(e):.1f}" cy="{y}" r="5" fill="#ffffff" stroke="{color}" stroke-width="2"/>')
                       + f'<rect x="{X(lo) - 6:.1f}" y="{y - 8}" width="{X(hi) - X(lo) + 12:.1f}" height="16" fill="#ffffff" fill-opacity="0"/></g>')
            out.append(f'<text x="{X(hi) + 8:.1f}" y="{y + 4}" font-size="11" fill="{INK2}">{e:.1f}</text>')

    def share_txt(blk):
        s = blk["margin_drop_source"]["pooled"]["share_of_gap_from_sparsity"]
        lo, hi = s["ci95"]
        return f"{s['estimate']:.2f} [{lo:.2f}, {'∞' if hi is None else f'{hi:.2f}'}]"
    out.append(f'<text x="24" y="{ay0 + band * 4 + 54}" font-size="12" fill="{INK2}">Share of the PGD−FGSM gap explained by '
               f'sparsity: development {share_txt(dev_a)}; confirmation {share_txt(con_a)}</text>')

    # Panel B
    bx0, bx1, by0, by1 = 680, 1040, 140, 420
    vals = [(pooled(r, "rand"), pooled(r, "pgd")) for r in dev_g + con_g]
    vmax = max(max(a, b) for a, b in vals)
    vmin = min(0.0, min(min(a, b) for a, b in vals))
    top = float(int(vmax / 4 + 1) * 4)
    bot = float(int(vmin / 4 - (vmin < 0)) * 4) if vmin < 0 else 0.0

    def PX(v):
        return bx0 + (bx1 - bx0) * (v - bot) / (top - bot)

    def PY(v):
        return by1 - (by1 - by0) * (v - bot) / (top - bot)

    out.append(f'<text x="{bx0 - 40}" y="{ay0 - 22}" font-size="14" font-weight="600" fill="{INK}">B · Per group: PGD-3 subset vs matched random</text>')
    t = bot
    while t <= top + 1e-9:
        out.append(f'<line x1="{PX(t):.1f}" x2="{PX(t):.1f}" y1="{by0}" y2="{by1}" stroke="{GRID}"/>'
                   f'<line x1="{bx0}" x2="{bx1}" y1="{PY(t):.1f}" y2="{PY(t):.1f}" stroke="{GRID}"/>'
                   f'<text x="{PX(t):.1f}" y="{by1 + 16}" font-size="11" fill="{MUTED}" text-anchor="middle">{t:g}</text>'
                   f'<text x="{bx0 - 8}" y="{PY(t) + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{t:g}</text>')
        t += 4
    out.append(f'<line x1="{PX(bot):.1f}" y1="{PY(bot):.1f}" x2="{PX(top):.1f}" y2="{PY(top):.1f}" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    out.append(f'<text x="{PX(top) - 4:.1f}" y="{PY(top) + 16:.1f}" font-size="11" fill="{MUTED}" text-anchor="end" '
               f'transform="rotate(-38 {PX(top) - 4:.1f} {PY(top) + 16:.1f})">no selection (y = x)</text>')
    out.append(f'<text x="{(bx0 + bx1) / 2:.0f}" y="{by1 + 36}" font-size="12" fill="{INK2}" text-anchor="middle">'
               'random subset, mean of 8 seeds (margin drop)</text>')
    out.append(f'<text x="{bx0 - 36}" y="{(by0 + by1) / 2:.0f}" font-size="12" fill="{INK2}" text-anchor="middle" '
               f'transform="rotate(-90 {bx0 - 36} {(by0 + by1) / 2:.0f})">PGD-3 subset (margin drop)</text>')
    for g, color, filled, lab in ((dev_g, ORANGE, False, "Development"), (con_g, BLUE, True, con_label)):
        above = sum(pooled(r, "pgd") > pooled(r, "rand") for r in g)
        for r in g:
            a, b = pooled(r, "rand"), pooled(r, "pgd")
            tip = f"{lab} · {r['scene_id']}: random {a:.2f}, PGD-3 {b:.2f}"
            mark = (f'<circle cx="{PX(a):.1f}" cy="{PY(b):.1f}" r="4.5" fill="{color}" fill-opacity="0.85" stroke="#ffffff" stroke-width="1.5"/>'
                    if filled else f'<circle cx="{PX(a):.1f}" cy="{PY(b):.1f}" r="4" fill="none" stroke="{color}" stroke-width="1.5" stroke-opacity="0.8"/>')
            out.append(f'<g><title>{esc(tip)}</title>{mark}<circle cx="{PX(a):.1f}" cy="{PY(b):.1f}" r="9" fill="#ffffff" fill-opacity="0"/></g>')
        ytxt = by0 + (14 if filled else 32)
        out.append(f'<text x="{bx0 + 10}" y="{ytxt}" font-size="12" fill="{INK}">{lab.split(" (")[0]}: {above}/{len(g)} groups above y = x</text>')
    out.append(f'<text x="24" y="{H - 14}" font-size="11" fill="{MUTED}">Subsets are masks of FGSM\'s own rounded perturbation with exactly k = |PGD-3 support| RGB channel coordinates; '
               'k is matched per group and source. Hover a mark for values.</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    con_json = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "analysis" / "confirm_analysis.json"
    con_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "analysis" / "confirm_per_group.csv"
    dest = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "figures" / "figure1_selection.svg"
    label = sys.argv[4] if len(sys.argv) > 4 else "Held-out confirmation (pre-registered, Checkpoint 2.4)"
    dev_a = json.loads((HERE / "analysis" / "dev_analysis_confirm_schema.json").read_text())
    dev_g = rows(HERE / "analysis" / "dev_per_group.csv")
    svg = build(dev_a, dev_g, json.loads(con_json.read_text()), rows(con_csv), label)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(svg + "\n")
    print(dest)


if __name__ == "__main__":
    main()
