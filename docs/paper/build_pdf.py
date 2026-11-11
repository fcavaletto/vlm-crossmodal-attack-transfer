"""Build the manuscript PDF from MANUSCRIPT.md (standard library + headless Google Chrome).

    python docs/paper/build_pdf.py            # writes MANUSCRIPT.html and the named PDF
    CHROME=/path/to/chrome python docs/paper/build_pdf.py

The converter handles only the Markdown subset the manuscript uses: headings, paragraphs, rules,
nested bullet and numbered lists, pipe tables, one figure per image line, and inline bold, italic,
code and links. The figure is regenerated first, from the frozen held-out outputs.
"""

from __future__ import annotations

import html
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "MANUSCRIPT.md"
OUT_HTML = HERE / "MANUSCRIPT.html"
OUT_PDF = HERE / "Cavaletto_2026_Which-Coordinates-Not-How-Many_review-draft.pdf"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

CSS = """
@page { size: A4; margin: 20mm 21mm 20mm 21mm;
        @bottom-center { content: counter(page); font: 9pt Helvetica, Arial, sans-serif; color: #52514e; } }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: Charter, "Bitstream Charter", Georgia, "Times New Roman", serif; font-size: 10.5pt;
       line-height: 1.42; color: #0b0b0b; background: #ffffff; margin: 0; }
h1 { font-size: 17.5pt; line-height: 1.22; margin: 0 0 8pt; font-weight: 700; }
h2 { font-size: 12.5pt; margin: 15pt 0 5pt; font-weight: 700; break-after: avoid; }
p { margin: 0 0 6.5pt; orphans: 3; widows: 3; }
p:has(> strong:only-child) { break-after: avoid; }  /* bold run-in headings stay with their text */
ul, ol { margin: 0 0 6.5pt; padding-left: 17pt; }
li { margin: 0 0 2.5pt; }
li > ul, li > ol { margin: 2.5pt 0 0; }
li:has(> ul), li:has(> ol) { break-inside: avoid; }  /* a bullet stays with its sub-list */
hr { border: none; border-top: 0.6pt solid #c9c8c2; margin: 9pt 0; }
code { font-family: Menlo, "DejaVu Sans Mono", monospace; font-size: 8.6pt; background: #f2f1ed; padding: 0 2pt;
       border-radius: 2pt; }
a { color: #1d5fae; text-decoration: none; }
table { border-collapse: collapse; margin: 4pt 0 9pt; font-size: 9pt; line-height: 1.3; break-inside: avoid; }
th, td { border-bottom: 0.5pt solid #d8d7d1; padding: 3pt 7pt 3pt 0; text-align: left; vertical-align: top; }
th { border-bottom: 0.9pt solid #0b0b0b; font-weight: 700; }
figure { margin: 8pt 0 12pt; break-inside: avoid; }
figure img { width: 100%; display: block; }
figcaption { font-size: 9pt; line-height: 1.35; color: #2b2a28; margin-top: 5pt; }
.byline { font-size: 11pt; margin: 0 0 3pt; }
.meta { font-size: 9.5pt; color: #52514e; }
.abstract { background: #f7f6f3; padding: 8pt 11pt 3pt; border-radius: 3pt; font-size: 10pt; line-height: 1.38; }
"""


def inline(s: str) -> str:
    s = s.replace("\\|", "\x00")
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", s)
    s = re.sub(r"&lt;(https?://[^&\s]+)&gt;", r'<a href="\1">\1</a>', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    return s.replace("\x00", "|")


def cells(line: str) -> list[str]:
    line = line.strip().replace("\\|", "\x00")
    parts = line.strip("|").split("|")
    return [p.strip().replace("\x00", "\\|") for p in parts]


LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")


def render_list(lines: list[str]) -> str:
    out: list[str] = []
    stack: list[tuple[int, str]] = []
    for ln in lines:
        m = LIST_RE.match(ln)
        ind, kind, body = len(m.group(1)), ("ol" if m.group(2)[0].isdigit() else "ul"), m.group(3)
        while stack and ind < stack[-1][0]:
            out.append(f"</li></{stack.pop()[1]}>")
        if not stack or ind > stack[-1][0]:
            out.append(f"<{kind}>")
            stack.append((ind, kind))
        else:
            out.append("</li>")
        out.append(f"<li>{inline(body)}")
    while stack:
        out.append(f"</li></{stack.pop()[1]}>")
    return "".join(out)


def alt_text(src: str) -> str:
    """Use an SVG figure's own <desc> as the alt text, so the two descriptions cannot drift apart."""
    path = HERE / src
    if path.suffix != ".svg" or not path.exists():
        return ""
    m = re.search(r"<desc[^>]*>(.*?)</desc>", path.read_text(), re.S)
    return html.unescape(m.group(1)) if m else ""


def convert(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i, in_abstract = 0, False
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("# "):
            out.append(f"<h1>{inline(ln[2:])}</h1>")
            i += 1
        elif ln.startswith("## "):
            if in_abstract:
                out.append("</div>")
                in_abstract = False
            title = ln[3:].strip()
            out.append(f"<h2>{inline(title)}</h2>")
            if title == "Abstract":
                out.append('<div class="abstract">')
                in_abstract = True
            i += 1
        elif ln.strip() == "---":
            if in_abstract:
                out.append("</div>")
                in_abstract = False
            out.append("<hr>")
            i += 1
        elif ln.startswith("!["):
            m = re.match(r"^!\[(.*)\]\(([^)]+)\)\s*$", ln)
            out.append(f'<figure><img src="{m.group(2)}" alt="{html.escape(alt_text(m.group(2)))}">'
                       f'<figcaption>{inline(m.group(1))}</figcaption></figure>')
            i += 1
        elif ln.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            head, body = cells(block[0]), [cells(b) for b in block[2:]]
            t = ["<table><thead><tr>", *[f"<th>{inline(c)}</th>" for c in head], "</tr></thead><tbody>"]
            for r in body:
                t += ["<tr>", *[f"<td>{inline(c)}</td>" for c in r], "</tr>"]
            out.append("".join(t) + "</tbody></table>")
        elif LIST_RE.match(ln):
            block = []
            while i < len(lines) and LIST_RE.match(lines[i]):
                block.append(lines[i])
                i += 1
            out.append(render_list(block))
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not LIST_RE.match(lines[i]) \
                    and not lines[i].startswith(("#", "|", "![")) and lines[i].strip() != "---":
                para.append(lines[i].strip())
                i += 1
            text = " ".join(para)
            if not out or not out[-1].startswith("<h1>") or "byline" in "".join(out[-1:]):
                out.append(f"<p>{inline(text)}</p>")
            else:
                cls = "byline"
                out.append(f'<p class="{cls}">{inline(text)}</p>')
    if in_abstract:
        out.append("</div>")
    html_body = "\n".join(out)
    html_body = html_body.replace('<p>Review draft,', '<p class="meta">Review draft,', 1)
    title = re.search(r"<h1>(.*?)</h1>", html_body).group(1)
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{re.sub('<[^>]+>', '', title)}</title><style>{CSS}</style></head>\n<body>\n{html_body}\n</body></html>\n")


def main() -> None:
    subprocess.run([sys.executable, "-I", str(HERE / "figures" / "make_main_figure.py")], check=True)
    OUT_HTML.write_text(convert(SRC.read_text()))
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files", f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {OUT_HTML.name} and {OUT_PDF.name}")


if __name__ == "__main__":
    main()
