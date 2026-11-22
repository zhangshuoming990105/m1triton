"""CLI entrypoint for `python -m m1triton` and the bundled `m1triton-opt` wrapper."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import m1triton


def find_m1triton_opt() -> Path | None:
    """Locate the native `m1triton-opt` binary inside the package or repo build tree."""
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "bin" / "m1triton-opt",
        pkg_dir.parents[1] / "build" / "m1triton-opt",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    on_path = shutil.which("m1triton-opt")
    if on_path:
        resolved = Path(on_path).resolve()
        try:
            with resolved.open("rb") as fh:
                magic = fh.read(4)
            if magic != b"#!/" and not magic.startswith(b"#!"):
                return resolved
        except OSError:
            pass
    return None


def opt_main(argv: Sequence[str] | None = None) -> int:
    """Invoke the bundled native `m1triton-opt` optimizer/codegen binary."""
    args = list(sys.argv[1:] if argv is None else argv)
    opt_bin = find_m1triton_opt()
    if opt_bin is None:
        print(
            "m1triton-opt binary not found. Build or install m1triton via `pip install -e .` or `cmake --build build`.",
            file=sys.stderr,
        )
        return 1
    proc = subprocess.run([str(opt_bin), *args], check=False)
    return int(proc.returncode)


def _print_usage() -> None:
    print(
        f"m1triton {m1triton.__version__}\n"
        "Usage: python -m m1triton <command> [args...]\n\n"
        "Commands:\n"
        "  doctor [--json] [--no-smoke]   Check Apple Silicon Metal environment and run GPU smoke tests\n"
        "  opt [args...]                  Run the native m1triton-opt SSA IR / MSL tool\n"
        "  version                        Print the installed m1triton version"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_usage()
        return 0

    cmd, rest = args[0], args[1:]
    if cmd in {"version", "--version", "-V"}:
        print(f"m1triton {m1triton.__version__}")
        return 0
    if cmd == "doctor":
        from m1triton.doctor import main as doctor_main

        return doctor_main(rest)
    if cmd == "opt":
        return opt_main(rest)

    print(f"Unknown command: {cmd}", file=sys.stderr)
    _print_usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
