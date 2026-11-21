"""M1Triton: Apple M1 Max specialized Triton-compatible and co-designed compiler."""

from __future__ import annotations

import sys
import types
from pathlib import Path

_BUILD_PKG_DIR = Path(__file__).resolve().parents[2] / "build" / "m1triton"
if _BUILD_PKG_DIR.is_dir() and str(_BUILD_PKG_DIR) not in __path__:
    __path__.append(str(_BUILD_PKG_DIR))

from m1triton import language, runtime, tlx
from m1triton.autotune import Autotuner, Config, autotune
from m1triton.jit import CompiledKernel, JITFunction, jit
from m1triton.language import cdiv


def next_power_of_2(n: int) -> int:
    """Return the smallest power of two >= `n`."""
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _make_triton_compat_module() -> types.ModuleType:
    mod = types.ModuleType("m1triton.triton_compat")
    mod.__doc__ = "Drop-in Triton compatibility namespace (`import m1triton.triton_compat as triton`)."
    mod.jit = jit
    mod.autotune = autotune
    mod.Config = Config
    mod.cdiv = cdiv
    mod.next_power_of_2 = next_power_of_2
    mod.language = language
    mod.tl = language
    mod.tlx = tlx
    mod.runtime = runtime
    sys.modules.setdefault("m1triton.triton_compat", mod)
    return mod


triton_compat = _make_triton_compat_module()

__all__ = [
    "Autotuner",
    "CompiledKernel",
    "Config",
    "JITFunction",
    "autotune",
    "cdiv",
    "jit",
    "language",
    "next_power_of_2",
    "runtime",
    "tlx",
    "triton_compat",
]
__version__ = "0.1.0"
