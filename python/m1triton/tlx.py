"""M1Triton Co-Design DSL extensions and hardware specialization hints (`tlx`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StructureHint:
    """Matrix sparsity/symmetry hint lowered to `ir.StructureAttr`."""

    kind: str = "dense"
    diagonal_offset: int = 0
    is_unit_diagonal: bool = False


@dataclass(frozen=True)
class TileSpecHint:
    """Explicit M1 Max SIMD-group tile schedule hint lowered to `ir.TileSpecAttr`."""

    bm: int = 64
    bn: int = 64
    bk: int = 16
    num_simdgroups: int = 4
    double_buffer: bool = False
    split_k: int = 1
    swizzle_w: int = 8


@dataclass(frozen=True)
class VecWidthHint:
    """Memory vectorization width hint lowered to `ir.VecWidthAttr`."""

    width: int = 4


@dataclass(frozen=True)
class LayoutHint:
    """Tensor layout hint lowered to `ir.LayoutHintAttr`."""

    layout: str = "nchw"
    prefer_vec4: bool = True


@dataclass(frozen=True)
class PersistentGridHint:
    """Persistent threadgroup grid hint lowered to `ir.PersistentGridAttr`."""

    enabled: bool = True
    max_tgs_per_core: int = 2
    swizzle_w: int = 8


def _jit_only(name: str) -> Any:
    raise RuntimeError(f"tlx.{name} is a Co-Design DSL primitive and must be called inside an @m1triton.jit kernel.")


def structure(
    tensor_or_kind: Any = None,
    kind: str | None = None,
    *,
    diagonal_offset: int = 0,
    is_unit_diagonal: bool = False,
) -> StructureHint:
    """Annotate a matrix tile or kernel with algebraic structure metadata.

    Inside `@m1triton.jit`:
      - `tlx.structure(a, "upper_triangular")` annotates SSA value `a` and returns it.
      - `tlx.structure("upper_triangular")` attaches the structure hint to the kernel.
    Outside `@m1triton.jit`, returns a `StructureHint` object usable in `@m1triton.jit(hints=[...])`.
    """
    if kind is None and isinstance(tensor_or_kind, str):
        return StructureHint(
            kind=tensor_or_kind,
            diagonal_offset=int(diagonal_offset),
            is_unit_diagonal=bool(is_unit_diagonal),
        )
    if kind is not None:
        return StructureHint(
            kind=str(kind),
            diagonal_offset=int(diagonal_offset),
            is_unit_diagonal=bool(is_unit_diagonal),
        )
    return _jit_only("structure")


def tile_spec(
    bm: int = 64,
    bn: int = 64,
    bk: int = 16,
    *,
    num_simdgroups: int = 4,
    double_buffer: bool = False,
    split_k: int = 1,
    swizzle_w: int = 8,
) -> TileSpecHint:
    """Attach an explicit M1 Max tile/SIMD-group schedule hint."""
    return TileSpecHint(
        bm=int(bm),
        bn=int(bn),
        bk=int(bk),
        num_simdgroups=int(num_simdgroups),
        double_buffer=bool(double_buffer),
        split_k=int(split_k),
        swizzle_w=int(swizzle_w),
    )


def vec_width(width: int) -> VecWidthHint:
    """Attach an explicit load/store vectorization width hint (1, 2, or 4)."""
    return VecWidthHint(width=int(width))


def layout_hint(layout: str = "nchw", *, prefer_vec4: bool = True) -> LayoutHint:
    """Attach a memory layout hint (`nchw`, `nhwc`, `ncdhw`, `row_major`, `col_major`)."""
    return LayoutHint(layout=str(layout), prefer_vec4=bool(prefer_vec4))


def persistent_grid(
    enabled: bool = True,
    *,
    max_tgs_per_core: int = 2,
    swizzle_w: int = 8,
) -> PersistentGridHint:
    """Attach a persistent threadgroup scheduling hint for M1 Max's 32 GPU cores."""
    return PersistentGridHint(
        enabled=bool(enabled),
        max_tgs_per_core=int(max_tgs_per_core),
        swizzle_w=int(swizzle_w),
    )


def online_softmax(
    x_ptr: Any,
    out_ptr: Any = None,
    num_rows: Any = None,
    row_len: Any = None,
    *,
    axis: int = -1,
    log_softmax: bool = False,
) -> Any:
    return _jit_only("online_softmax")


def welford_norm(
    kind: str,
    *args: Any,
    eps: float = 1e-5,
    axis: int = -1,
    num_groups: int = 1,
    stage: int = 0,
    has_weight: bool = False,
    has_bias: bool = False,
) -> Any:
    return _jit_only("welford_norm")


def pool(
    x_ptr: Any,
    out_ptr: Any,
    *dims: Any,
    kind: str = "max",
    ndim: int = 2,
    kernel_size: tuple[int, ...] | list[int] | int = 2,
    stride: tuple[int, ...] | list[int] | int = 2,
    padding: tuple[int, ...] | list[int] | int = 0,
    dilation: tuple[int, ...] | list[int] | int = 1,
    count_include_pad: bool = True,
    ceil_mode: bool = False,
) -> Any:
    return _jit_only("pool")


def row_scan(
    x_ptr: Any,
    out_ptr: Any,
    num_rows: Any,
    row_len: Any,
    *,
    op: str = "add",
    exclusive: bool = False,
    reverse: bool = False,
    mask_ptr: Any = None,
) -> Any:
    return _jit_only("row_scan")


def simd_scan(
    x: Any,
    *,
    op: str = "add",
    axis: int = -1,
    exclusive: bool = False,
    reverse: bool = False,
    mask: Any = None,
) -> Any:
    return _jit_only("simd_scan")


def implicit_gemm_conv(
    x: Any,
    weight: Any,
    *,
    stride: tuple[int, ...] = (1, 1),
    padding: tuple[int, ...] = (0, 0),
    dilation: tuple[int, ...] = (1, 1),
    groups: int = 1,
    layout: str = "nchw",
) -> Any:
    return _jit_only("implicit_gemm_conv")


def flash_attention_tile(
    q: Any,
    k: Any,
    v: Any,
    *,
    scale: float = 1.0,
    causal: bool = False,
    causal_offset: int = 0,
) -> Any:
    return _jit_only("flash_attention_tile")


__all__ = [
    "StructureHint",
    "TileSpecHint",
    "VecWidthHint",
    "LayoutHint",
    "PersistentGridHint",
    "structure",
    "tile_spec",
    "vec_width",
    "layout_hint",
    "persistent_grid",
    "online_softmax",
    "welford_norm",
    "pool",
    "row_scan",
    "simd_scan",
    "implicit_gemm_conv",
    "flash_attention_tile",
]
