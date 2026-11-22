"""Triton-compatible language markers and dtypes (`import m1triton.language as tl`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class dtype:
    """Scalar element dtype descriptor matching M1Triton SSA IR ScalarDType."""

    name: str
    itemsize: int
    is_floating: bool
    is_signed: bool
    is_bool: bool
    ir_enum_name: str

    def __repr__(self) -> str:
        return f"m1triton.language.{self.name}"


float32 = dtype("float32", 4, True, True, False, "F32")
float16 = dtype("float16", 2, True, True, False, "F16")
int32 = dtype("int32", 4, False, True, False, "I32")
uint32 = dtype("uint32", 4, False, False, False, "U32")
int64 = dtype("int64", 8, False, True, False, "I64")
bool_ = dtype("bool_", 1, False, False, True, "Bool")
int1 = bool_


class constexpr:
    """Compile-time specialization parameter marker (`BLOCK_SIZE: tl.constexpr`)."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, constexpr):
            return type(self.value) is type(other.value) and self.value == other.value
        return False

    def __hash__(self) -> int:
        return hash((type(self.value), self.value))

    def __repr__(self) -> str:
        return f"constexpr({self.value!r})"


def cdiv(x: int, y: int) -> int:
    """Ceiling integer division `(x + y - 1) // y`."""
    return (x + y - 1) // y


def _jit_only(name: str) -> Any:
    raise RuntimeError(f"tl.{name} is a symbolic DSL primitive and must be called inside an @m1triton.jit kernel.")


def static_range(*args: Any, **kwargs: Any) -> Any:
    return _jit_only("static_range")


def program_id(axis: int) -> Any:
    return _jit_only("program_id")


def num_programs(axis: int) -> Any:
    return _jit_only("num_programs")


def arange(start: int, end: int) -> Any:
    return _jit_only("arange")


def zeros(shape: tuple[int, ...] | list[int], dtype: dtype = float32) -> Any:
    return _jit_only("zeros")


def full(shape: tuple[int, ...] | list[int], value: Any, dtype: dtype | None = None) -> Any:
    return _jit_only("full")


def load(
    pointer: Any,
    mask: Any = None,
    other: Any = None,
    boundary_check: tuple[int, ...] = (),
    padding_option: str = "",
    eviction_policy: str = "",
    volatile: bool = False,
) -> Any:
    return _jit_only("load")


def store(
    pointer: Any,
    value: Any,
    mask: Any = None,
    boundary_check: tuple[int, ...] = (),
    eviction_policy: str = "",
) -> Any:
    return _jit_only("store")


def atomic_add(pointer: Any, val: Any, mask: Any = None, sem: str | None = None, scope: str | None = None) -> Any:
    return _jit_only("atomic_add")


def dot(
    input: Any,
    other: Any,
    acc: Any = None,
    allow_tf32: bool = False,
    out_dtype: dtype = float32,
    input_precision: str | None = None,
) -> Any:
    return _jit_only("dot")


def sum(input: Any, axis: int | None = None, keep_dims: bool = False) -> Any:
    return _jit_only("sum")


def max(input: Any, axis: int | None = None, keep_dims: bool = False, return_indices: bool = False) -> Any:
    return _jit_only("max")


def min(input: Any, axis: int | None = None, keep_dims: bool = False, return_indices: bool = False) -> Any:
    return _jit_only("min")


def argmax(input: Any, axis: int, keep_dims: bool = False) -> Any:
    return _jit_only("argmax")


def argmin(input: Any, axis: int, keep_dims: bool = False) -> Any:
    return _jit_only("argmin")


def cumsum(input: Any, axis: int = 0, reverse: bool = False) -> Any:
    return _jit_only("cumsum")


def cumprod(input: Any, axis: int = 0, reverse: bool = False) -> Any:
    return _jit_only("cumprod")


def where(condition: Any, x: Any, y: Any) -> Any:
    return _jit_only("where")


def cast(input: Any, dtype: dtype) -> Any:
    return _jit_only("cast")


def trans(input: Any, *dims: int) -> Any:
    return _jit_only("trans")


def reshape(input: Any, shape: tuple[int, ...] | list[int]) -> Any:
    return _jit_only("reshape")


def broadcast_to(input: Any, shape: tuple[int, ...] | list[int]) -> Any:
    return _jit_only("broadcast_to")


def expand_dims(input: Any, axis: int) -> Any:
    return _jit_only("expand_dims")


def exp(x: Any) -> Any:
    return _jit_only("exp")


def log(x: Any) -> Any:
    return _jit_only("log")


def sqrt(x: Any) -> Any:
    return _jit_only("sqrt")


def rsqrt(x: Any) -> Any:
    return _jit_only("rsqrt")


def abs(x: Any) -> Any:
    return _jit_only("abs")


def sigmoid(x: Any) -> Any:
    return _jit_only("sigmoid")


def tanh(x: Any) -> Any:
    return _jit_only("tanh")


def erf(x: Any) -> Any:
    return _jit_only("erf")


def floor(x: Any) -> Any:
    return _jit_only("floor")


def ceil(x: Any) -> Any:
    return _jit_only("ceil")


def sin(x: Any) -> Any:
    return _jit_only("sin")


def cos(x: Any) -> Any:
    return _jit_only("cos")


def maximum(x: Any, y: Any) -> Any:
    return _jit_only("maximum")


def minimum(x: Any, y: Any) -> Any:
    return _jit_only("minimum")


def multiple_of(input: Any, values: Any) -> Any:
    return _jit_only("multiple_of")


def max_contiguous(input: Any, values: Any) -> Any:
    return _jit_only("max_contiguous")


def static_assert(cond: Any, msg: str = "") -> None:
    if not cond:
        raise AssertionError(msg or "tl.static_assert failed")


__all__ = [
    "dtype",
    "constexpr",
    "float32",
    "float16",
    "int32",
    "uint32",
    "int64",
    "bool_",
    "int1",
    "cdiv",
    "static_range",
    "program_id",
    "num_programs",
    "arange",
    "zeros",
    "full",
    "load",
    "store",
    "atomic_add",
    "dot",
    "sum",
    "max",
    "min",
    "argmax",
    "argmin",
    "cumsum",
    "cumprod",
    "where",
    "cast",
    "trans",
    "reshape",
    "broadcast_to",
    "expand_dims",
    "exp",
    "log",
    "sqrt",
    "rsqrt",
    "abs",
    "sigmoid",
    "tanh",
    "erf",
    "floor",
    "ceil",
    "sin",
    "cos",
    "maximum",
    "minimum",
    "multiple_of",
    "max_contiguous",
    "static_assert",
]
