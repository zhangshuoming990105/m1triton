"""M1Triton: Apple M1 Max specialized Triton-compatible and co-designed compiler."""

from pathlib import Path

_BUILD_PKG_DIR = Path(__file__).resolve().parents[2] / "build" / "m1triton"
if _BUILD_PKG_DIR.is_dir() and str(_BUILD_PKG_DIR) not in __path__:
    __path__.append(str(_BUILD_PKG_DIR))

from m1triton import runtime

__all__ = ["runtime"]
__version__ = "0.1.0"
