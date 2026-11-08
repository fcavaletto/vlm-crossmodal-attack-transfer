"""Download the pinned Qwen files into HF_HOME and check every sha256.

Run on the pod before `measure`. The weights are not in the image: the pod
fetches exactly the files listed in qwen_files.sha256, at the pinned revision,
and exits non-zero unless every file matches. `--verify-only` skips the download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ID = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
MANIFEST = Path(__file__).resolve().with_name("qwen_files.sha256")


def read_manifest(path: Path = MANIFEST) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest
    return expected


def snapshot_dir(home: Path) -> Path:
    return home / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / REVISION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(home: Path, expected: dict[str, str]) -> dict[str, object]:
    folder = snapshot_dir(home)
    files = {}
    for name, want in sorted(expected.items()):
        path = folder / name
        got = sha256(path) if path.is_file() else None
        files[name] = {"expected": want, "actual": got, "match": got == want}
    return {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "snapshot": str(folder),
        "files": files,
        "all_match": all(entry["match"] for entry in files.values()),
    }


def download(home: Path, names: list[str]) -> None:
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import hf_hub_download

    for name in names:
        hf_hub_download(REPO_ID, name, revision=REVISION, cache_dir=str(home / "hub"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path(os.environ.get("HF_HOME", "/models")))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    expected = read_manifest()
    if not args.verify_only:
        download(args.home, sorted(expected))
    report = verify(args.home, expected)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0 if report["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
