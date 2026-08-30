"""Download the real archival datasets Phase 3 is anchored on.

    python scripts/fetch_data.py [--sources ednet_kt1,assistments_2009] [--force]

Every source is downloaded to `data/raw/`, verified against a pinned SHA-256,
and recorded in `data/raw/manifest.json`. A source that fails to download or
fails verification prints manual download instructions and the script exits
non-zero. **There is no synthetic fallback.** Silently substituting generated
data for real data is the failure mode this whole phase exists to avoid.

URLs are pinned to an immutable revision (a git commit or a Hugging Face
dataset commit) so a re-run downloads the same bytes the digest was taken from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"

CHUNK = 1 << 20
PROGRESS_EVERY = 32 * CHUNK

# Pinned mirrors. The canonical hosts for these datasets are a Google Sites page
# (ASSISTments) and a Google Drive folder (EdNet); neither serves a stable,
# scriptable URL, so each source below points at an immutable mirror and the
# `manual` text names the canonical source a human should cite and check.
SOURCES: dict[str, dict] = {
    "assistments_2009": {
        "filename": "assistments-2009-2010-skill-builder.csv",
        "url": (
            "https://raw.githubusercontent.com/CAHLR/pyBKT-examples/"
            "f5a0fcd41eaeb3d17639d78aaa51198b9f05ba0a/data/as.csv"
        ),
        "sha256": "f22e3fb7872c1784ce93b0f9ebabbe0cbcac4f896fd8b4a11667b9715d77dbdc",
        "bytes": 83201940,
        "role": "primary",
        "description": "ASSISTments 2009-2010 skill builder: attempts, hints, skills, first-response time.",
        "licence": "Released for research use by the ASSISTments project (Worcester Polytechnic Institute).",
        "citation": "Feng, Heffernan & Koedinger (2009). Addressing the assessment challenge with an online system that tutors as it assesses.",
        "manual": (
            "Download 'skill_builder_data.csv' from the ASSISTments data page\n"
            "  https://sites.google.com/site/assistmentsdata/datasets/2009-2010-assistment-data\n"
            "and save it as data/raw/assistments-2009-2010-skill-builder.csv"
        ),
    },
    "assistments_2012": {
        "filename": "assistments-2012-2013.zip",
        "url": (
            "https://huggingface.co/datasets/Lucy9999/assist2012/resolve/"
            "ffbc1d3370e8749b9f97c81fc8810543ffb7c81f/"
            "2012-2013-data-with-predictions-4-final.zip"
        ),
        "sha256": "7bb18f932f97e63c2eb34bb869c44cd4f46f4adec4d7fe02298578dbba572445",
        "bytes": 576365408,
        "role": "secondary",
        "description": "ASSISTments 2012-2013 with affect predictions: larger, with start/end timestamps and response time.",
        "licence": "Released for research use by the ASSISTments project; mirror published under MIT.",
        "citation": "Feng, Heffernan & Koedinger (2009); 2012-2013 school-year release.",
        "manual": (
            "Download '2012-2013-data-with-predictions-4-final.zip' from\n"
            "  https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect\n"
            "and save it as data/raw/assistments-2012-2013.zip"
        ),
    },
    "ednet_kt1": {
        "filename": "ednet-kt1-shard0.parquet",
        "url": (
            "https://huggingface.co/datasets/mgor/EDNet/resolve/"
            "2e0fb7f2f0dbcb8cd62a29bb30d9e89ec943a145/kt1/train-00000-of-00011.parquet"
        ),
        "sha256": "7908e8d37925de5321877a7a62bfe082aa7cda38b415923e2b28e350fe61fdd9",
        "bytes": 170673611,
        "role": "scale",
        "description": (
            "EdNet KT1, shard 0 of 11 (8.66M interactions): elapsed time at scale. "
            "The full release is ~131M interactions; BUILD.md Phase 3 caps this source "
            "at ~5M, so only one shard is fetched and prep_archival subsamples it further."
        ),
        "licence": "CC BY-NC 4.0 (Riiid Labs).",
        "citation": "Choi et al. (2020). EdNet: A Large-Scale Hierarchical Dataset in Education.",
        "manual": (
            "Download kt1/train-00000-of-00011.parquet from\n"
            "  https://huggingface.co/datasets/mgor/EDNet\n"
            "or reconstruct it from the original release at https://github.com/riiid/ednet\n"
            "and save it as data/raw/ednet-kt1-shard0.parquet"
        ),
    },
    "ednet_questions": {
        "filename": "ednet-questions.parquet",
        "url": (
            "https://huggingface.co/datasets/mgor/EDNet/resolve/"
            "2e0fb7f2f0dbcb8cd62a29bb30d9e89ec943a145/questions/train-00000-of-00001.parquet"
        ),
        "sha256": "90a63786bac294e1e9e20b6f78ab2ee787934159d95c3669c79768c69645f84d",
        "bytes": 311363,
        "role": "metadata",
        "description": "EdNet question metadata: knowledge-component tags, TOEIC part, correct answer.",
        "licence": "CC BY-NC 4.0 (Riiid Labs).",
        "citation": "Choi et al. (2020).",
        "manual": (
            "Download questions/train-00000-of-00001.parquet from\n"
            "  https://huggingface.co/datasets/mgor/EDNet\n"
            "and save it as data/raw/ednet-questions.parquet"
        ),
    },
}

# Junyi Academy is listed as optional in BUILD.md Phase 3 and is not fetched:
# its prerequisite structure duplicates what the bank's own 36-concept DAG
# already provides, and three sources already cover the attempt/timing/scale
# axes the study needs.


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            sha.update(block)
    return sha.hexdigest()


def download(url: str, target: Path) -> None:
    """Stream to a .part file, then move into place. Partial files never win."""
    part = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "alp-system/fetch_data"})
    with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        seen = next_report = 0
        while block := response.read(CHUNK):
            handle.write(block)
            seen += len(block)
            if seen >= next_report:
                pct = f" ({100 * seen / total:.0f}%)" if total else ""
                print(f"  {seen / 1e6:.0f} MB{pct}", flush=True)
                next_report = seen + PROGRESS_EVERY
    shutil.move(part, target)


def fail(name: str, source: dict, reason: str) -> None:
    print(f"\nFAILED: {name} — {reason}", file=sys.stderr)
    print(f"\nManual download instructions for {name}:\n", file=sys.stderr)
    print(source["manual"], file=sys.stderr)
    print(
        f"\nThen re-run this script; it verifies SHA-256 {source['sha256']}"
        "\nand will not proceed on a mismatch. Do not substitute synthetic data.",
        file=sys.stderr,
    )


def fetch(name: str, source: dict, force: bool) -> dict:
    target = RAW_DIR / source["filename"]

    if target.exists() and not force:
        print(f"{name}: {target.relative_to(ROOT)} present, verifying")
    else:
        print(f"{name}: downloading {source['url']}")
        try:
            download(source["url"], target)
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            fail(name, source, f"download failed: {error}")
            raise SystemExit(1)

    found = digest(target)
    if found != source["sha256"]:
        fail(name, source, f"SHA-256 mismatch\n  expected {source['sha256']}\n  found    {found}")
        raise SystemExit(1)

    size = target.stat().st_size
    print(f"{name}: verified {target.relative_to(ROOT)} ({size / 1e6:.1f} MB, sha256 ok)")
    return {
        "source": name,
        "file": str(target.relative_to(ROOT)),
        "url": source["url"],
        "sha256": found,
        "bytes": size,
        "role": source["role"],
        "description": source["description"],
        "licence": source["licence"],
        "citation": source["citation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        default=",".join(SOURCES),
        help=f"comma-separated subset of {','.join(SOURCES)}",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if the file is present")
    parser.add_argument("--seed", type=int, default=20260821, help="recorded in the manifest; nothing here is random")
    args = parser.parse_args()

    names = [name.strip() for name in args.sources.split(",") if name.strip()]
    unknown = [name for name in names if name not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown source(s): {', '.join(unknown)}. Known: {', '.join(SOURCES)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    entries = [fetch(name, SOURCES[name], args.force) for name in names]

    manifest = {
        "seed": args.seed,
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {MANIFEST_PATH.relative_to(ROOT)} ({len(entries)} sources)")


if __name__ == "__main__":
    main()
