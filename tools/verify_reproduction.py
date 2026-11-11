"""Model-free reproduction check. Needs Python, numpy and Pillow. No model, no GPU, no network.

    python tools/verify_reproduction.py

It writes only to a temporary directory and never touches a committed file. Checks:

1. Frozen protocol: every file in docs/checkpoint-2.4-confirm/FROZEN.json hashes to its
   frozen value, FROZEN.json itself is unchanged since the freeze commit, and no frozen file
   was committed after the freeze.
2. Held-out split: ``python -m scene_confirm`` regenerates fixtures/trackB_confirm_v1
   (needs the recorded Pillow, 13.0.0; the check says so if another is installed).
   On macOS, the study platform, every file must be byte-identical. Elsewhere PNG bytes depend on
   the zlib build, so each PNG must instead decode to identical RGB uint8 pixels, size and
   metadata, and every other file must still be byte-identical.
3. Committed analysis outputs: the frozen confirmation analysis, Figure 1 and the Checkpoint 3
   pilot analysis are re-run from committed records and compared with the committed files.
   The per-group CSV, the figure and the pilot JSON must be byte-identical. The confirmation
   JSON must be identical except its ``input`` field, which records an absolute path.

No tolerance is applied anywhere. Any difference is a failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "docs" / "checkpoint-2.4-confirm"
PILOT = ROOT / "docs" / "checkpoint-3-mechanism"
SPLIT = ROOT / "fixtures" / "trackB_confirm_v1"
FREEZE_COMMIT = "c35b02c"
RECORDED_PILLOW = "13.0.0"
STUDY_PLATFORM = "darwin"  # Apple silicon macOS, where the split was generated

failures: list[str] = []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ok: bool, label: str, detail: str = "") -> None:
    print(("  ok    " if ok else "  FAIL  ") + label + (f": {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def run(*args: str) -> None:
    env = dict(os.environ)
    src = sorted(str(p) for p in (ROOT / "src").iterdir() if p.is_dir())
    env["PYTHONPATH"] = os.pathsep.join(src + [env.get("PYTHONPATH", "")])
    done = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, env=env)
    if done.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(args)}\n{done.stdout}\n{done.stderr}")


def frozen_protocol() -> None:
    print("1. Frozen protocol (Checkpoint 2.4)")
    frozen = json.loads((CONFIRM / "FROZEN.json").read_text(encoding="utf-8"))["files"]
    bad = [name for name, digest in frozen.items() if sha256(ROOT / name) != digest]
    check(not bad, f"{len(frozen)} frozen files match FROZEN.json", ", ".join(bad))
    if git("cat-file", "-e", f"{FREEZE_COMMIT}^{{commit}}").returncode != 0:
        check(False, "freeze commit is in the history", f"{FREEZE_COMMIT} not found (shallow clone?)")
        return
    at_freeze = git("show", f"{FREEZE_COMMIT}:docs/checkpoint-2.4-confirm/FROZEN.json").stdout
    check(at_freeze == (CONFIRM / "FROZEN.json").read_text(encoding="utf-8"), "FROZEN.json is unchanged since the freeze commit")
    later = git("log", "--format=%h", f"{FREEZE_COMMIT}..HEAD", "--", "docs/checkpoint-2.4-confirm/FROZEN.json", *frozen).stdout.split()
    check(not later, "no frozen file was committed after the freeze", " ".join(later))


def decoded(path: Path):
    """Decode a PNG without converting it: format, mode, size, metadata and the raw array."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        arr = np.asarray(im)
        return im.format, im.mode, im.size, dict(im.info), arr


def held_out_split(tmp: Path) -> None:
    print("2. Held-out split regeneration")
    import numpy as np
    import PIL

    out = tmp / "confirm_split"
    run("-m", "scene_confirm", "--output", str(out), "--root", str(ROOT))
    listed = json.loads((SPLIT / "checksums.json").read_text(encoding="utf-8"))["files"]
    if PIL.__version__ != RECORDED_PILLOW:
        check(False, "recorded Pillow is installed", f"found {PIL.__version__}, the split was built with {RECORDED_PILLOW}")
    stale = [name for name, info in listed.items() if sha256(SPLIT / name) != info["sha256"]]
    check(not stale, f"every committed split file matches the committed checksums.json ({len(listed)} files)", ", ".join(stale[:5]))
    committed = sorted(str(p.relative_to(SPLIT)) for p in SPLIT.rglob("*") if p.is_file())
    regenerated = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    check(committed == regenerated, "regenerated file list equals the committed one")
    common = [name for name in committed if (out / name).is_file()]
    pngs = [name for name in common if name.endswith(".png")]
    others = [name for name in common if not name.endswith(".png") and name != "checksums.json"]

    if sys.platform == STUDY_PLATFORM:
        # The platform the split was built on: everything must be byte-identical, PNGs included.
        bad = [name for name, info in listed.items() if not (out / name).is_file() or sha256(out / name) != info["sha256"]]
        check(not bad, f"{len(listed)} regenerated files match checksums.json", ", ".join(bad[:5]))
        differing = [name for name in common if (out / name).read_bytes() != (SPLIT / name).read_bytes()]
        check(not differing, "every regenerated file is byte-identical to the committed one", ", ".join(differing[:5]))
        if differing:
            describe_differences(out, differing)
        return

    # Other platforms. PNG compression output depends on the zlib build in the Pillow wheel
    # (docs/PUBLICATION_AUDIT.md section 6), so PNG bytes are not compared here. Everything else
    # still is, and every PNG must decode to the same format, mode, size, metadata, dtype, shape
    # and pixel values. No tolerance.
    print(f"  note  platform {sys.platform}: PNG files are compared by exact decoded pixels; "
          f"byte-exact PNG comparison runs on {STUDY_PLATFORM}")
    differing = [name for name in others if (out / name).read_bytes() != (SPLIT / name).read_bytes()]
    check(not differing, f"every regenerated non-PNG file is byte-identical ({len(others)} files: manifest, metadata, pixel hashes)",
          ", ".join(differing))
    new_sums = json.loads((out / "checksums.json").read_text(encoding="utf-8"))
    same_keys = new_sums.get("algorithm") == "sha256" and set(new_sums["files"]) == set(listed)
    non_png_same = same_keys and all(new_sums["files"][n] == listed[n] for n in listed if not n.endswith(".png"))
    check(same_keys and non_png_same, "regenerated checksums.json lists the same files, with identical entries for non-PNG files")
    mismatched = []
    for name in pngs:
        fa, ma, sa, ia, xa = decoded(out / name)
        fb, mb, sb, ib, xb = decoded(SPLIT / name)
        same = (fa, ma, sa, ia, xa.dtype, xa.shape) == (fb, mb, sb, ib, xb.dtype, xb.shape) and fa == "PNG" and ma == "RGB"
        if not (same and xa.dtype == np.uint8 and np.array_equal(xa, xb)):
            mismatched.append(name)
    check(not mismatched,
          f"every regenerated PNG decodes to identical RGB uint8 pixels, size and metadata ({len(pngs)} files)",
          ", ".join(mismatched[:5]))
    if mismatched:
        describe_differences(out, mismatched)


def describe_differences(out: Path, differing: list[str]) -> None:
    """Diagnostic only. It prints what differs and never turns a failure into a pass."""
    import numpy as np
    import PIL.features
    from PIL import Image

    print(f"  info  platform {sys.platform}, zlib {PIL.features.version('zlib')}, "
          f"freetype {PIL.features.version('freetype2')}, {len(differing)} file(s) differ")
    same_pixels, worst = 0, []
    pngs = [name for name in differing if name.endswith(".png")]
    for name in pngs:
        with Image.open(out / name) as a, Image.open(SPLIT / name) as b:
            if a.mode != b.mode or a.size != b.size:
                worst.append((name, "mode or size differs", 0))
                continue
            x, y = np.asarray(a).astype(np.int16), np.asarray(b).astype(np.int16)
        if np.array_equal(x, y):
            same_pixels += 1
        else:
            worst.append((name, int(np.abs(x - y).max()), int((x != y).any(axis=-1).sum()) if x.ndim == 3 else int((x != y).sum())))
    print(f"  info  PNG files with different bytes: {len(pngs)}; of these, decoded pixels identical: {same_pixels}")
    for name, peak, count in worst[:10]:
        print(f"  info  pixels differ in {name}: largest channel difference {peak}, pixels affected {count}")
    for name in [n for n in differing if not n.endswith(".png")][:5]:
        print(f"  info  non-image file differs: {name}")


def analyses(tmp: Path) -> None:
    print("3. Committed analysis outputs and Figure 1")
    out_json = tmp / "confirm_analysis.json"
    run(str(CONFIRM / "analyze_confirm.py"), str(CONFIRM / "run" / "records.jsonl"), str(out_json))
    new = json.loads(out_json.read_text(encoding="utf-8"))
    old = json.loads((CONFIRM / "analysis" / "confirm_analysis.json").read_text(encoding="utf-8"))
    check(set(new) == set(old), "confirmation analysis has the same fields")
    differing = sorted(key for key in old if new.get(key) != old[key])
    check(differing in ([], ["input"]), "confirmation analysis is identical apart from the recorded input path", ", ".join(differing))
    check(
        (tmp / "confirm_per_group.csv").read_bytes() == (CONFIRM / "analysis" / "confirm_per_group.csv").read_bytes(),
        "per-group table is byte-identical",
    )
    figure = tmp / "figure1_selection.svg"
    run(str(CONFIRM / "make_figure.py"), str(out_json), str(tmp / "confirm_per_group.csv"), str(figure))
    check(figure.read_bytes() == (CONFIRM / "figures" / "figure1_selection.svg").read_bytes(), "Figure 1 is byte-identical")
    pilot = tmp / "pilot_analysis.json"
    run(str(PILOT / "analyze_pilot.py"), str(PILOT / "pilot"), str(pilot))
    check(pilot.read_bytes() == (PILOT / "analysis" / "pilot_analysis.json").read_bytes(), "Checkpoint 3 pilot analysis is byte-identical")


def main() -> None:
    before = git("status", "--porcelain").stdout
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        frozen_protocol()
        held_out_split(tmp)
        analyses(tmp)
    check(git("status", "--porcelain").stdout == before, "the working tree was not modified by this check")
    if failures:
        raise SystemExit(f"\n{len(failures)} check(s) failed.")
    print("\nAll reproduction checks passed.")


if __name__ == "__main__":
    main()
