#!/usr/bin/env python3
"""Generate concise Markdown and CLI scoreboards from frozen eval_kernelbench.py JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# M1Triton KernelBench Level 1 Scoreboard",
        "",
        f"- KernelBench Commit: `{report.get('kernelbench_commit_sha', 'unknown')[:12]}`",
        f"- Mode: `{report.get('mode', 'unknown')}` | Seeds: `{report.get('seeds')}` | Warmup: `{report.get('warmup')}` | Trials: `{report.get('trials')}`",
        f"- Passed: **{summary['passed']}/{summary['total_evaluated']}** | Beats Baseline: **{summary['beats_baseline']}/{summary['total_evaluated']}** | Geomean Speedup: **{summary['geomean_speedup']}x**",
        f"- PyTorch MPS Unsupported (CPU fallback baseline): `{summary.get('mps_unsupported_task_ids', [])}`",
        "",
        "| ID | Task | Status | Backend | Base (ms) | M1Triton (ms) | Speedup | Dispatches |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for t in report["tasks"]:
        base_ms = f"{t['baseline_median_ms']:.4f}" if "baseline_median_ms" in t else "-"
        cand_ms = f"{t['candidate_median_ms']:.4f}" if "candidate_median_ms" in t else "-"
        sp = f"{t['speedup']:.2f}x" if "speedup" in t else "-"
        lines.append(
            f"| {t['id']} | `{t['name']}` | {t['status']} | `{t['baseline_backend']}` | "
            f"{base_ms} | {cand_ms} | {sp} | {t.get('compiled_kernel_dispatch_count', 0)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render KernelBench scoreboard from eval JSON.")
    parser.add_argument("eval_json", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top-slowest", type=int, default=15)
    args = parser.parse_args()

    report = json.loads(args.eval_json.read_text(encoding="utf-8"))
    md = render_markdown(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"Wrote scoreboard to {args.out}")
    else:
        print(md)

    passed_with_sp = [t for t in report["tasks"] if t.get("passed") and t.get("speedup")]
    if passed_with_sp:
        slowest = sorted(passed_with_sp, key=lambda x: x["speedup"])[: args.top_slowest]
        print(f"\nSlowest {len(slowest)} passing tasks (lowest speedup):")
        for t in slowest:
            print(
                f"  #{t['id']:3d} {t['name']:<44} base={t['baseline_median_ms']:.3f}ms "
                f"cand={t['candidate_median_ms']:.3f}ms speedup={t['speedup']:.2f}x"
            )


if __name__ == "__main__":
    main()
