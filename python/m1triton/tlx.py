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


@dataclass(frozen=True)
class EpilogueStepHint:
    """Single step in a composable register-level epilogue DAG."""

    op: str
    param0: float = 0.0
    param1: float = 0.0
    arg_idx: int = -1


@dataclass(frozen=True)
class EpilogueDAGHint:
    """Composable register-level post-op DAG lowered to `ir.EpilogueDAGAttr`."""

    steps: tuple[EpilogueStepHint, ...] = ()


@dataclass(frozen=True)
class FusedTailHint:
    """Producer-consumer reduction or pooling tail lowered to `ir.FusedTailAttr`."""

    kind: str = "none"
    kernel_size: int = 1
    stride: int = 1
    post_scale: float = 1.0
    keep_dims: bool = True


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


def epilogue(steps: Any) -> EpilogueDAGHint:
    """Attach a composable epilogue DAG (`[("relu",), ("add_const", 1.0), ("mul_bias", ptr), ...]`)."""
    param_ref_ops = {
        "add_bias", "sub_bias", "mul_bias", "div_bias", "min_bias", "max_bias",
        "add_scalar", "sub_scalar", "mul_scalar", "div_scalar",
        "add_residual", "sub_residual", "mul_residual",
        "save_reg", "add_saved", "sub_saved", "mul_saved",
    }
    parsed: list[EpilogueStepHint] = []
    for raw in steps:
        if isinstance(raw, EpilogueStepHint):
            parsed.append(raw)
        elif isinstance(raw, str):
            op_key = raw.strip().lower().replace("-", "_")
            if op_key == "softplus":
                parsed.append(EpilogueStepHint(op=raw, param0=1.0, param1=20.0))
            else:
                parsed.append(EpilogueStepHint(op=raw))
        elif isinstance(raw, (tuple, list)) and len(raw) >= 1:
            op = str(raw[0])
            op_key = op.strip().lower().replace("-", "_")
            if op_key in param_ref_ops:
                if len(raw) < 2:
                    raise ValueError(f"tlx.epilogue {op!r} requires a parameter/slot operand")
                if not isinstance(raw[1], int):
                    raise TypeError(
                        f"tlx.epilogue {op!r} outside @jit body requires integer param/slot index, got {raw[1]!r}"
                    )
                parsed.append(EpilogueStepHint(op=op, param0=0.0, param1=0.0, arg_idx=int(raw[1])))
            elif op_key == "softplus":
                p0 = float(raw[1]) if len(raw) >= 2 and raw[1] is not None else 1.0
                p1 = float(raw[2]) if len(raw) >= 3 and raw[2] is not None else 20.0
                parsed.append(EpilogueStepHint(op=op, param0=p0, param1=p1, arg_idx=-1))
            else:
                p0 = float(raw[1]) if len(raw) >= 2 and raw[1] is not None else 0.0
                p1 = float(raw[2]) if len(raw) >= 3 and raw[2] is not None else 0.0
                aidx = int(raw[3]) if len(raw) >= 4 and raw[3] is not None else -1
                parsed.append(EpilogueStepHint(op=op, param0=p0, param1=p1, arg_idx=aidx))
        else:
            raise TypeError(f"Unsupported epilogue step spec: {raw!r}")
    return EpilogueDAGHint(steps=tuple(parsed))


def fused_tail(
    kind: str,
    *,
    kernel_size: int = 1,
    stride: int = 1,
    post_scale: float = 1.0,
    keep_dims: bool = True,
) -> FusedTailHint:
    """Attach a producer-consumer reduction or pooling tail hint."""
    return FusedTailHint(
        kind=str(kind),
        kernel_size=int(kernel_size),
        stride=int(stride),
        post_scale=float(post_scale),
        keep_dims=bool(keep_dims),
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
    x_ptr: Any,
    w_ptr: Any,
    out_ptr: Any,
    *args: Any,
    ndim: int = 2,
    kernel_size: tuple[int, ...] | list[int] | int = 3,
    stride: tuple[int, ...] | list[int] | int = 1,
    padding: tuple[int, ...] | list[int] | int = 0,
    dilation: tuple[int, ...] | list[int] | int = 1,
    output_padding: tuple[int, ...] | list[int] | int = 0,
    groups: int = 1,
    transposed: bool = False,
    has_bias: bool = False,
    is_depthwise: bool = False,
    is_pointwise: bool = False,
    layout: str = "nchw",
) -> Any:
    return _jit_only("implicit_gemm_conv")


def fused_loss(
    kind: str,
    *args: Any,
    margin: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-6,
    stage: int = 1,
) -> Any:
    return _jit_only("fused_loss")


def flash_attention_tile(
    q_ptr: Any,
    k_ptr: Any,
    v_ptr: Any,
    out_ptr: Any,
    bh: Any = None,
    lq: Any = None,
    lkv: Any = None,
    d: Any = None,
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
    "EpilogueStepHint",
    "EpilogueDAGHint",
    "FusedTailHint",
    "structure",
    "tile_spec",
    "vec_width",
    "layout_hint",
    "persistent_grid",
    "epilogue",
    "fused_tail",
    "online_softmax",
    "welford_norm",
    "pool",
    "row_scan",
    "simd_scan",
    "implicit_gemm_conv",
    "fused_loss",
    "flash_attention_tile",
]
