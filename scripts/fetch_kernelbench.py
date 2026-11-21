#!/usr/bin/env python3
"""Fetch ScalingIntelligence/KernelBench Level 1 tasks (1..100) at a pinned commit SHA."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REPO = "https://github.com/ScalingIntelligence/KernelBench.git"
DEFAULT_COMMIT = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_task_id(filename: str) -> int | None:
    match = re.match(r"^(\d+)_.*\.py$", filename)
    return int(match.group(1)) if match else None


def fetch_level1(dest_dir: Path, repo_url: str, commit_sha: str) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kb_fetch_") as tmp:
        tmp_repo = Path(tmp) / "KernelBench"
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", repo_url, str(tmp_repo)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_repo), "fetch", "--depth", "1", "origin", commit_sha],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_repo), "sparse-checkout", "init", "--cone"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_repo), "sparse-checkout", "set", "KernelBench/level1"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_repo), "checkout", commit_sha],
            check=True,
        )
        resolved_sha = subprocess.check_output(
            ["git", "-C", str(tmp_repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if resolved_sha != commit_sha:
            raise RuntimeError(f"Expected commit {commit_sha}, got {resolved_sha}")

        src_dir = tmp_repo / "KernelBench" / "level1"
        src_files = sorted(
            [p for p in src_dir.glob("*.py") if parse_task_id(p.name) is not None],
            key=lambda p: parse_task_id(p.name) or 0,
        )
        if len(src_files) != 100:
            raise RuntimeError(f"Expected 100 Level 1 tasks, found {len(src_files)}")

        for old in dest_dir.glob("*.py"):
            old.unlink()

        tasks = []
        for src in src_files:
            tid = parse_task_id(src.name)
            assert tid is not None
            dst = dest_dir / src.name
            shutil.copy2(src, dst)
            tasks.append(
                {
                    "id": tid,
                    "filename": src.name,
                    "name": src.stem[len(str(tid)) + 1 :].strip("_"),
                    "sha256": sha256_file(dst),
                }
            )

    ids = [t["id"] for t in tasks]
    if ids != list(range(1, 101)):
        raise RuntimeError(f"Task IDs are not contiguous 1..100: {ids[:10]}...{ids[-10:]}")

    manifest = {
        "source_repo": repo_url,
        "commit_sha": commit_sha,
        "level": 1,
        "task_count": len(tasks),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks,
    }
    manifest_path = dest_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch pinned KernelBench Level 1 tasks.")
    parser.add_argument("--dest", type=Path, default=Path("benchmarks/kernelbench_l1"))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    args = parser.parse_args()

    manifest = fetch_level1(args.dest, args.repo, args.commit)
    print(
        f"Fetched {manifest['task_count']} KernelBench Level 1 tasks "
        f"(commit {manifest['commit_sha'][:12]}) -> {args.dest}"
    )


if __name__ == "__main__":
    main()
