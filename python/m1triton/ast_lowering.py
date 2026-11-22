"""Python AST to M1Triton C++ SSA IR lowering (`@m1triton.jit` / `@triton.jit`)."""

from __future__ import annotations

import ast
import inspect
import operator
import textwrap
from dataclasses import dataclass
from typing import Any, Sequence

from m1triton import _C
from m1triton import language as tl
from m1triton import tlx

_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class ParamSpec:
    """Specialized parameter signature passed from the JIT launcher into AST lowering."""

    name: str
    is_constexpr: bool
    constexpr_val: Any = None
    is_pointer: bool = False
    dtype: tl.dtype = tl.float32
    divisibility: int = 1


@dataclass(frozen=True)
class _SSAVal:
    """Tracks an emitted C++ SSA ValueId along with pointer provenance and structure hints."""

    vid: int
    ptr_params: frozenset[int] = frozenset()
    structure: Any = None


_PY_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

_PY_CMPOPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_FLOAT_ONLY_UNARY_OPS = frozenset(
    {"exp", "log", "sqrt", "rsqrt", "sigmoid", "tanh", "erf", "floor", "ceil", "sin", "cos"}
)


def parse_kernel_ast(fn: Any) -> tuple[ast.FunctionDef, str]:
    """Parse a Python `@jit` function into its dedented source and `ast.FunctionDef`."""
    raw_src = inspect.getsource(fn)
    src = textwrap.dedent(raw_src)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node, src
    raise ValueError(f"Could not locate function definition in source of {fn!r}")


def to_ir_scalar_dtype(dt: tl.dtype | Any) -> _C.ir.ScalarDType:
    """Convert a `tl.dtype` or `_C.ir.ScalarDType` to `_C.ir.ScalarDType`."""
    ir = _C.ir
    if isinstance(dt, ir.ScalarDType):
        return dt
    if isinstance(dt, tl.dtype):
        return getattr(ir.ScalarDType, dt.ir_enum_name)
    raise TypeError(f"Unsupported M1Triton dtype: {dt!r}")


def from_ir_scalar_dtype(dt: _C.ir.ScalarDType) -> tl.dtype:
    """Convert `_C.ir.ScalarDType` back to a `tl.dtype`."""
    ir = _C.ir
    mapping = {
        ir.ScalarDType.F32: tl.float32,
        ir.ScalarDType.F16: tl.float16,
        ir.ScalarDType.I32: tl.int32,
        ir.ScalarDType.U32: tl.uint32,
        ir.ScalarDType.I64: tl.int64,
        ir.ScalarDType.Bool: tl.bool_,
    }
    return mapping[dt]


def _scalar_dtype_bytes(dt: _C.ir.ScalarDType) -> int:
    return from_ir_scalar_dtype(dt).itemsize


def _is_float_dtype(dt: _C.ir.ScalarDType) -> bool:
    return dt in (_C.ir.ScalarDType.F32, _C.ir.ScalarDType.F16)


def _is_int_dtype(dt: _C.ir.ScalarDType) -> bool:
    return dt in (_C.ir.ScalarDType.I32, _C.ir.ScalarDType.U32, _C.ir.ScalarDType.I64)


def _promote_scalar_dtype(a: _C.ir.ScalarDType, b: _C.ir.ScalarDType) -> _C.ir.ScalarDType:
    ir = _C.ir
    if a == b:
        return a
    if a == ir.ScalarDType.Bool:
        return b
    if b == ir.ScalarDType.Bool:
        return a
    if ir.ScalarDType.F32 in (a, b):
        return ir.ScalarDType.F32
    if ir.ScalarDType.F16 in (a, b):
        return ir.ScalarDType.F16
    if ir.ScalarDType.I64 in (a, b):
        return ir.ScalarDType.I64
    if ir.ScalarDType.U32 in (a, b):
        return ir.ScalarDType.U32
    return ir.ScalarDType.I32


def _broadcast_shapes(shape_a: Sequence[int], shape_b: Sequence[int]) -> list[int]:
    if not shape_a:
        return list(shape_b)
    if not shape_b:
        return list(shape_a)
    rank = max(len(shape_a), len(shape_b))
    pad_a = [1] * (rank - len(shape_a)) + list(shape_a)
    pad_b = [1] * (rank - len(shape_b)) + list(shape_b)
    out: list[int] = []
    for da, db in zip(pad_a, pad_b):
        if da == db:
            out.append(da)
        elif da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        else:
            raise ValueError(f"Incompatible shapes for broadcasting: {list(shape_a)} and {list(shape_b)}")
    return out


def _parse_structure_attr(
    kind: str, diagonal_offset: int = 0, is_unit_diagonal: bool = False
) -> _C.ir.StructureAttr:
    ir = _C.ir
    key = kind.strip().lower().replace("-", "_")
    mapping = {
        "dense": ir.MatrixStructure.Dense,
        "upper_triangular": ir.MatrixStructure.UpperTriangular,
        "upper": ir.MatrixStructure.UpperTriangular,
        "triu": ir.MatrixStructure.UpperTriangular,
        "lower_triangular": ir.MatrixStructure.LowerTriangular,
        "lower": ir.MatrixStructure.LowerTriangular,
        "tril": ir.MatrixStructure.LowerTriangular,
        "symmetric": ir.MatrixStructure.Symmetric,
        "sym": ir.MatrixStructure.Symmetric,
        "diagonal": ir.MatrixStructure.Diagonal,
        "diag": ir.MatrixStructure.Diagonal,
        "causal_mask": ir.MatrixStructure.CausalMask,
        "causal": ir.MatrixStructure.CausalMask,
    }
    if key not in mapping:
        raise ValueError(f"Unknown tlx.structure kind: {kind!r}")
    return ir.StructureAttr(mapping[key], int(diagonal_offset), bool(is_unit_diagonal))


def _parse_layout_hint(layout: str) -> _C.ir.TensorLayoutHint:
    ir = _C.ir
    key = layout.strip().lower().replace("-", "_")
    mapping = {
        "nchw": ir.TensorLayoutHint.NCHW,
        "nhwc": ir.TensorLayoutHint.NHWC,
        "ncdhw": ir.TensorLayoutHint.NCDHW,
        "row_major": ir.TensorLayoutHint.RowMajor,
        "rowmajor": ir.TensorLayoutHint.RowMajor,
        "col_major": ir.TensorLayoutHint.ColMajor,
        "colmajor": ir.TensorLayoutHint.ColMajor,
    }
    if key not in mapping:
        raise ValueError(f"Unknown tlx.layout_hint layout: {layout!r}")
    return mapping[key]


def _unwrap_const(val: Any) -> Any:
    return val.value if isinstance(val, tl.constexpr) else val


def _parse_epilogue_op(op: str) -> _C.ir.EpilogueOpKind:
    ir = _C.ir
    key = op.strip().lower().replace("-", "_")
    mapping = {
        "add_const": ir.EpilogueOpKind.AddConst,
        "sub_const": ir.EpilogueOpKind.SubConst,
        "rsub_const": ir.EpilogueOpKind.RSubConst,
        "mul_const": ir.EpilogueOpKind.MulConst,
        "div_const": ir.EpilogueOpKind.DivConst,
        "min_const": ir.EpilogueOpKind.MinConst,
        "max_const": ir.EpilogueOpKind.MaxConst,
        "clamp": ir.EpilogueOpKind.Clamp,
        "add_bias": ir.EpilogueOpKind.AddBias,
        "sub_bias": ir.EpilogueOpKind.SubBias,
        "mul_bias": ir.EpilogueOpKind.MulBias,
        "div_bias": ir.EpilogueOpKind.DivBias,
        "min_bias": ir.EpilogueOpKind.MinBias,
        "max_bias": ir.EpilogueOpKind.MaxBias,
        "add_scalar": ir.EpilogueOpKind.AddScalar,
        "sub_scalar": ir.EpilogueOpKind.SubScalar,
        "mul_scalar": ir.EpilogueOpKind.MulScalar,
        "div_scalar": ir.EpilogueOpKind.DivScalar,
        "add_residual": ir.EpilogueOpKind.AddResidual,
        "sub_residual": ir.EpilogueOpKind.SubResidual,
        "mul_residual": ir.EpilogueOpKind.MulResidual,
        "save_reg": ir.EpilogueOpKind.SaveReg,
        "add_saved": ir.EpilogueOpKind.AddSaved,
        "sub_saved": ir.EpilogueOpKind.SubSaved,
        "mul_saved": ir.EpilogueOpKind.MulSaved,
        "relu": ir.EpilogueOpKind.ReLU,
        "leaky_relu": ir.EpilogueOpKind.LeakyReLU,
        "sigmoid": ir.EpilogueOpKind.Sigmoid,
        "tanh": ir.EpilogueOpKind.Tanh,
        "gelu": ir.EpilogueOpKind.GELU,
        "gelu_tanh": ir.EpilogueOpKind.GELUTanh,
        "swish": ir.EpilogueOpKind.Swish,
        "silu": ir.EpilogueOpKind.Swish,
        "mish": ir.EpilogueOpKind.Mish,
        "hardswish": ir.EpilogueOpKind.HardSwish,
        "hardsigmoid": ir.EpilogueOpKind.HardSigmoid,
        "hardtanh": ir.EpilogueOpKind.HardTanh,
        "elu": ir.EpilogueOpKind.ELU,
        "selu": ir.EpilogueOpKind.SELU,
        "softplus": ir.EpilogueOpKind.Softplus,
        "softsign": ir.EpilogueOpKind.Softsign,
        "log": ir.EpilogueOpKind.Log,
        "exp": ir.EpilogueOpKind.Exp,
        "abs": ir.EpilogueOpKind.Abs,
        "neg": ir.EpilogueOpKind.Neg,
        "sqrt": ir.EpilogueOpKind.Sqrt,
    }
    if key not in mapping:
        raise ValueError(f"Unknown tlx.epilogue op: {op!r}")
    return mapping[key]


def _parse_tail_kind(kind: str) -> _C.ir.TailKind:
    ir = _C.ir
    key = kind.strip().lower().replace("-", "_")
    mapping = {
        "none": getattr(ir.TailKind, "None"),
        "row_sum": ir.TailKind.RowSum,
        "row_mean": ir.TailKind.RowMean,
        "row_max": ir.TailKind.RowMax,
        "row_min": ir.TailKind.RowMin,
        "row_logsumexp": ir.TailKind.RowLogSumExp,
        "spatial_global_avg_pool": ir.TailKind.SpatialGlobalAvgPool,
        "spatial_global_sum": ir.TailKind.SpatialGlobalSum,
        "spatial_global_max": ir.TailKind.SpatialGlobalMax,
        "pool1d_max": ir.TailKind.Pool1DMax,
        "pool1d_avg": ir.TailKind.Pool1DAvg,
    }
    if key not in mapping:
        raise ValueError(f"Unknown tlx.fused_tail kind: {kind!r}")
    return mapping[key]


def _apply_kernel_hint(kernel: _C.ir.Kernel, hint: Any) -> None:
    ir = _C.ir
    hints = kernel.hints
    if isinstance(hint, tlx.StructureHint):
        hints.structure = _parse_structure_attr(hint.kind, hint.diagonal_offset, hint.is_unit_diagonal)
    elif isinstance(hint, tlx.TileSpecHint):
        ts = ir.TileSpecAttr()
        ts.bm = int(hint.bm)
        ts.bn = int(hint.bn)
        ts.bk = int(hint.bk)
        ts.num_simdgroups = int(hint.num_simdgroups)
        ts.double_buffer = bool(hint.double_buffer)
        ts.split_k = int(hint.split_k)
        ts.swizzle_w = int(hint.swizzle_w)
        hints.tile_spec = ts
    elif isinstance(hint, tlx.VecWidthHint):
        hints.vec_width = ir.VecWidthAttr(int(hint.width))
    elif isinstance(hint, tlx.LayoutHint):
        lh = ir.LayoutHintAttr()
        lh.layout = _parse_layout_hint(hint.layout)
        lh.prefer_vec4 = bool(hint.prefer_vec4)
        hints.layout_hint = lh
    elif isinstance(hint, tlx.PersistentGridHint):
        pg = ir.PersistentGridAttr()
        pg.enabled = bool(hint.enabled)
        pg.max_tgs_per_core = int(hint.max_tgs_per_core)
        pg.swizzle_w = int(hint.swizzle_w)
        hints.persistent_grid = pg
    elif isinstance(hint, tlx.EpilogueDAGHint):
        dag = ir.EpilogueDAGAttr()
        dag.steps = [
            ir.EpilogueStep(_parse_epilogue_op(s.op), float(s.param0), float(s.param1), int(s.arg_idx))
            for s in hint.steps
        ]
        hints.epilogue_dag = dag
    elif isinstance(hint, tlx.FusedTailHint):
        ft = ir.FusedTailAttr()
        ft.kind = _parse_tail_kind(hint.kind)
        ft.kernel_size = int(hint.kernel_size)
        ft.stride = int(hint.stride)
        ft.post_scale = float(hint.post_scale)
        ft.keep_dims = bool(hint.keep_dims)
        hints.fused_tail = ft
    else:
        raise TypeError(f"Unsupported M1Triton kernel hint: {hint!r}")
    kernel.hints = hints


def _collect_target_names(target: ast.AST, out: list[str], seen: set[str]) -> None:
    if isinstance(target, ast.Name):
        if target.id not in seen:
            seen.add(target.id)
            out.append(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _collect_target_names(elt, out, seen)


def _collect_assigned_names(stmts: Sequence[ast.stmt]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def walk(body: Sequence[ast.stmt]) -> None:
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    _collect_target_names(tgt, out, seen)
            elif isinstance(stmt, ast.AugAssign):
                _collect_target_names(stmt.target, out, seen)
            elif isinstance(stmt, ast.AnnAssign):
                _collect_target_names(stmt.target, out, seen)
            elif isinstance(stmt, ast.For):
                _collect_target_names(stmt.target, out, seen)
                walk(stmt.body)
                walk(stmt.orelse)
            elif isinstance(stmt, ast.If):
                walk(stmt.body)
                walk(stmt.orelse)
            elif isinstance(stmt, ast.With):
                walk(stmt.body)

    walk(stmts)
    return out


def _is_ssa_or_numeric(val: Any) -> bool:
    return isinstance(val, (_SSAVal, bool, int, float))


class _ASTIRLowerer(ast.NodeVisitor):
    """Lowers a Python kernel `ast.FunctionDef` directly into C++ `IRBuilder` SSA ops."""

    def __init__(
        self,
        fn: Any,
        func_def: ast.FunctionDef,
        param_specs: Sequence[ParamSpec],
        kernel_name: str,
        initial_hints: Sequence[Any] = (),
    ) -> None:
        self.fn = fn
        self.func_def = func_def
        self.module = _C.ir.Module()
        self.kernel = self.module.add_kernel(kernel_name)
        self.builder = _C.ir.IRBuilder(self.kernel)

        self.env: dict[str, Any] = {}
        self.param_vals: dict[str, _SSAVal] = {}
        self._seed_globals()
        self._register_params(param_specs)
        for hint in initial_hints:
            _apply_kernel_hint(self.kernel, hint)

    def _seed_globals(self) -> None:
        fn_globals = getattr(self.fn, "__globals__", None) or {}
        self.env.update(fn_globals)
        closure = getattr(self.fn, "__closure__", None)
        code = getattr(self.fn, "__code__", None)
        if closure and code:
            for name, cell in zip(code.co_freevars, closure):
                try:
                    self.env[name] = cell.cell_contents
                except ValueError:
                    pass

    def _register_params(self, param_specs: Sequence[ParamSpec]) -> None:
        ir = _C.ir
        for ps in param_specs:
            param_idx = len(self.kernel.params)
            if ps.is_constexpr:
                val = ps.constexpr_val
                if isinstance(val, tl.constexpr):
                    val = val.value
                self.env[ps.name] = val
                if isinstance(val, bool):
                    self.builder.add_param(
                        ps.name,
                        ir.Type.scalar(ir.ScalarDType.Bool),
                        is_constexpr=True,
                        constexpr_val=int(val),
                    )
                elif isinstance(val, int):
                    dt = ir.ScalarDType.I32 if _INT32_MIN <= val <= _INT32_MAX else ir.ScalarDType.I64
                    self.builder.add_param(
                        ps.name,
                        ir.Type.scalar(dt),
                        is_constexpr=True,
                        constexpr_val=int(val),
                    )
            elif ps.is_pointer:
                ty = ir.Type.ptr(to_ir_scalar_dtype(ps.dtype), ir.AddrSpace.Device)
                vid = self.builder.add_param(
                    ps.name,
                    ty,
                    is_constexpr=False,
                    constexpr_val=0,
                    divisibility=int(ps.divisibility),
                    writable=False,
                )
                val = _SSAVal(vid=vid, ptr_params=frozenset({param_idx}))
                self.env[ps.name] = val
                self.param_vals[ps.name] = val
            else:
                ty = ir.Type.scalar(to_ir_scalar_dtype(ps.dtype))
                vid = self.builder.add_param(
                    ps.name,
                    ty,
                    is_constexpr=False,
                    constexpr_val=0,
                    divisibility=int(ps.divisibility),
                    writable=False,
                )
                val = _SSAVal(vid=vid)
                self.env[ps.name] = val
                self.param_vals[ps.name] = val

    def lower(self) -> _C.ir.Module:
        for stmt in self.func_def.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue
            self.visit(stmt)
        if not self.kernel.block_has_terminator(self.builder.current_block()):
            self.builder.emit_return()
        return self.module

    # --- SSA conversion, coercion, and broadcasting helpers ---

    def _emit_const(self, val: bool | int | float, dt: _C.ir.ScalarDType) -> _SSAVal:
        ir = _C.ir
        if dt == ir.ScalarDType.Bool:
            return _SSAVal(self.builder.emit_const_bool(bool(val)))
        if dt == ir.ScalarDType.I32:
            return _SSAVal(self.builder.emit_const_i32(int(val)))
        if dt == ir.ScalarDType.U32:
            return _SSAVal(self.builder.emit_const_u32(int(val)))
        if dt == ir.ScalarDType.I64:
            return _SSAVal(self.builder.emit_const_i64(int(val)))
        if dt == ir.ScalarDType.F16:
            return _SSAVal(self.builder.emit_const_f16(float(val)))
        if dt == ir.ScalarDType.F32:
            return _SSAVal(self.builder.emit_const_f32(float(val)))
        raise TypeError(f"Unsupported constant dtype: {dt!r}")

    def _materialize_ssa(self, val: Any, hint_dtype: _C.ir.ScalarDType | None = None) -> _SSAVal:
        ir = _C.ir
        if isinstance(val, _SSAVal):
            return val
        if isinstance(val, tl.constexpr):
            val = val.value
        if isinstance(val, bool):
            dt = hint_dtype if hint_dtype is not None else ir.ScalarDType.Bool
            return self._emit_const(val, dt)
        if isinstance(val, int):
            if hint_dtype is not None:
                return self._emit_const(val, hint_dtype)
            dt = ir.ScalarDType.I32 if _INT32_MIN <= val <= _INT32_MAX else ir.ScalarDType.I64
            return self._emit_const(val, dt)
        if isinstance(val, float):
            if hint_dtype is not None and _is_float_dtype(hint_dtype):
                return self._emit_const(val, hint_dtype)
            return self._emit_const(val, ir.ScalarDType.F32)
        raise TypeError(f"Cannot convert compile-time value {val!r} to an SSA IR value")

    def _coerce_to_dtype(self, val: Any, target_dt: _C.ir.ScalarDType) -> _SSAVal:
        if not isinstance(val, _SSAVal):
            return self._emit_const(val, target_dt)
        cur_ty = self.kernel.value_type(val.vid)
        if cur_ty.is_ptr_elem():
            raise TypeError(f"Cannot cast pointer type {cur_ty} to scalar dtype {target_dt}")
        cur_dt = cur_ty.scalar_elem()
        if cur_dt == target_dt:
            return val
        return _SSAVal(
            vid=self.builder.emit_cast(val.vid, target_dt),
            ptr_params=val.ptr_params,
            structure=val.structure,
        )

    def _broadcast_to_shape(self, val: _SSAVal, target_shape: Sequence[int]) -> _SSAVal:
        target = [int(d) for d in target_shape]
        ty = self.kernel.value_type(val.vid)
        if not target:
            if ty.is_tile():
                raise ValueError(f"Cannot broadcast tile {ty} down to scalar")
            return val
        if ty.is_scalar() or ty.is_ptr():
            vid = self.builder.emit_splat(val.vid, target)
            return _SSAVal(vid=vid, ptr_params=val.ptr_params, structure=val.structure)
        cur_shape = list(ty.shape())
        if cur_shape == target:
            return val
        cur = val
        while len(cur_shape) < len(target):
            cur = _SSAVal(
                vid=self.builder.emit_expand_dims(cur.vid, 0),
                ptr_params=cur.ptr_params,
                structure=cur.structure,
            )
            cur_shape = [1] + cur_shape
        if cur_shape != target:
            cur = _SSAVal(
                vid=self.builder.emit_broadcast(cur.vid, target),
                ptr_params=cur.ptr_params,
                structure=cur.structure,
            )
        return cur

    def _ensure_bool(self, val: Any) -> _SSAVal:
        ir = _C.ir
        if not isinstance(val, _SSAVal):
            return self._emit_const(bool(val), ir.ScalarDType.Bool)
        ty = self.kernel.value_type(val.vid)
        if ty.is_bool_elem():
            return val
        dt = ty.scalar_elem()
        zero = self._emit_const(0, dt)
        if ty.is_tile():
            zero = self._broadcast_to_shape(zero, ty.shape())
        return _SSAVal(self.builder.emit_binary(ir.BinaryOp.CmpNe, val.vid, zero.vid))

    def _lower_ptr_add(self, ptr_val: _SSAVal, off_raw: Any) -> _SSAVal:
        ir = _C.ir
        ptr_ty = self.kernel.value_type(ptr_val.vid)
        if isinstance(off_raw, _SSAVal):
            off_val = off_raw
        else:
            dt = ir.ScalarDType.I32 if _INT32_MIN <= int(off_raw) <= _INT32_MAX else ir.ScalarDType.I64
            off_val = self._emit_const(int(off_raw), dt)
        off_ty = self.kernel.value_type(off_val.vid)
        if not off_ty.is_int_elem():
            off_val = self._coerce_to_dtype(off_val, ir.ScalarDType.I32)
            off_ty = self.kernel.value_type(off_val.vid)

        if ptr_ty.is_ptr():
            vid = self.builder.emit_ptr_add(ptr_val.vid, off_val.vid)
            return _SSAVal(vid=vid, ptr_params=ptr_val.ptr_params)

        target_shape = _broadcast_shapes(ptr_ty.shape(), off_ty.shape())
        p_b = self._broadcast_to_shape(ptr_val, target_shape)
        o_b = self._broadcast_to_shape(off_val, target_shape)
        vid = self.builder.emit_ptr_add(p_b.vid, o_b.vid)
        return _SSAVal(vid=vid, ptr_params=ptr_val.ptr_params)

    def _align_binary_operands(
        self, lhs: Any, rhs: Any, *, force_float: bool = False
    ) -> tuple[_SSAVal, _SSAVal, _C.ir.ScalarDType]:
        ir = _C.ir
        if isinstance(lhs, _SSAVal) and not isinstance(rhs, _SSAVal):
            lhs_ty = self.kernel.value_type(lhs.vid)
            lhs_dt = lhs_ty.scalar_elem()
            if isinstance(rhs, float) and _is_int_dtype(lhs_dt):
                target_dt = ir.ScalarDType.F32
            elif force_float and not _is_float_dtype(lhs_dt):
                target_dt = ir.ScalarDType.F32
            else:
                target_dt = lhs_dt
            lhs_ssa = self._coerce_to_dtype(lhs, target_dt)
            rhs_ssa = self._emit_const(rhs, target_dt)
        elif not isinstance(lhs, _SSAVal) and isinstance(rhs, _SSAVal):
            rhs_ty = self.kernel.value_type(rhs.vid)
            rhs_dt = rhs_ty.scalar_elem()
            if isinstance(lhs, float) and _is_int_dtype(rhs_dt):
                target_dt = ir.ScalarDType.F32
            elif force_float and not _is_float_dtype(rhs_dt):
                target_dt = ir.ScalarDType.F32
            else:
                target_dt = rhs_dt
            lhs_ssa = self._emit_const(lhs, target_dt)
            rhs_ssa = self._coerce_to_dtype(rhs, target_dt)
        else:
            lhs_ssa = self._materialize_ssa(lhs)
            rhs_ssa = self._materialize_ssa(rhs)
            dt_a = self.kernel.value_type(lhs_ssa.vid).scalar_elem()
            dt_b = self.kernel.value_type(rhs_ssa.vid).scalar_elem()
            target_dt = _promote_scalar_dtype(dt_a, dt_b)
            if force_float and not _is_float_dtype(target_dt):
                target_dt = ir.ScalarDType.F32
            lhs_ssa = self._coerce_to_dtype(lhs_ssa, target_dt)
            rhs_ssa = self._coerce_to_dtype(rhs_ssa, target_dt)

        shape_a = self.kernel.value_type(lhs_ssa.vid).shape()
        shape_b = self.kernel.value_type(rhs_ssa.vid).shape()
        target_shape = _broadcast_shapes(shape_a, shape_b)
        if target_shape:
            lhs_ssa = self._broadcast_to_shape(lhs_ssa, target_shape)
            rhs_ssa = self._broadcast_to_shape(rhs_ssa, target_shape)
        return lhs_ssa, rhs_ssa, target_dt

    def _apply_binop(self, op_node: ast.operator, lhs: Any, rhs: Any) -> Any:
        if not isinstance(lhs, _SSAVal) and not isinstance(rhs, _SSAVal):
            py_fn = _PY_BINOPS.get(type(op_node))
            if py_fn is None:
                raise NotImplementedError(f"Unsupported binary operator: {type(op_node).__name__}")
            return py_fn(lhs, rhs)

        ir = _C.ir
        if isinstance(op_node, ast.Add):
            if isinstance(lhs, _SSAVal) and self.kernel.value_type(lhs.vid).is_ptr_elem():
                return self._lower_ptr_add(lhs, rhs)
            if isinstance(rhs, _SSAVal) and self.kernel.value_type(rhs.vid).is_ptr_elem():
                return self._lower_ptr_add(rhs, lhs)
        if isinstance(op_node, ast.Sub):
            if isinstance(lhs, _SSAVal) and self.kernel.value_type(lhs.vid).is_ptr_elem():
                neg_rhs = _SSAVal(self.builder.emit_unary(ir.UnaryOp.Neg, rhs.vid)) if isinstance(rhs, _SSAVal) else -rhs
                return self._lower_ptr_add(lhs, neg_rhs)

        is_true_div = isinstance(op_node, ast.Div)
        is_floor_div = isinstance(op_node, ast.FloorDiv)
        lhs_ssa, rhs_ssa, dt = self._align_binary_operands(lhs, rhs, force_float=is_true_div)

        op_map = {
            ast.Add: ir.BinaryOp.Add,
            ast.Sub: ir.BinaryOp.Sub,
            ast.Mult: ir.BinaryOp.Mul,
            ast.Div: ir.BinaryOp.Div,
            ast.FloorDiv: ir.BinaryOp.Div,
            ast.Mod: ir.BinaryOp.Rem,
            ast.Pow: ir.BinaryOp.Pow,
            ast.BitAnd: ir.BinaryOp.And,
            ast.BitOr: ir.BinaryOp.Or,
            ast.BitXor: ir.BinaryOp.Xor,
            ast.LShift: ir.BinaryOp.Shl,
            ast.RShift: ir.BinaryOp.Shr,
        }
        ir_op = op_map.get(type(op_node))
        if ir_op is None:
            raise NotImplementedError(f"Unsupported SSA binary operator: {type(op_node).__name__}")
        res_vid = self.builder.emit_binary(ir_op, lhs_ssa.vid, rhs_ssa.vid)
        if is_floor_div and _is_float_dtype(dt):
            res_vid = self.builder.emit_unary(ir.UnaryOp.Floor, res_vid)
        return _SSAVal(res_vid)

    # --- Statement visitors ---

    def _bind_target(self, target: ast.AST, value: Any) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = value
            if isinstance(value, _SSAVal):
                self.kernel.set_value_debug_name(value.vid, target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, (tuple, list)) or len(value) != len(target.elts):
                raise ValueError(f"Cannot unpack {value!r} into {len(target.elts)} targets")
            for elt, v in zip(target.elts, value):
                self._bind_target(elt, v)
        else:
            raise NotImplementedError(f"Unsupported assignment target: {type(target).__name__}")

    def visit_Assign(self, node: ast.Assign) -> None:
        val = self._eval(node.value)
        for target in node.targets:
            self._bind_target(target, val)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            return
        val = self._eval(node.value)
        self._bind_target(node.target, val)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("Augmented assignment only supports simple variable names")
        name = node.target.id
        if name not in self.env:
            raise NameError(f"Undefined variable in augmented assignment: {name}")
        cur = self.env[name]

        if (
            isinstance(node.op, ast.Add)
            and isinstance(cur, _SSAVal)
            and isinstance(node.value, ast.Call)
            and self._is_tl_dot_without_acc(node.value)
        ):
            args, kwargs = self._eval_call_args(node.value)
            kwargs["acc"] = cur
            res = self._lower_tl_dot(args, kwargs)
            self._bind_target(node.target, res)
            return

        rhs = self._eval(node.value)
        res = self._apply_binop(node.op, cur, rhs)
        self._bind_target(node.target, res)

    def _is_tl_dot_without_acc(self, call_node: ast.Call) -> bool:
        func = call_node.func
        is_dot = False
        if isinstance(func, ast.Attribute) and func.attr == "dot" and isinstance(func.value, ast.Name):
            owner = self.env.get(func.value.id)
            if owner is tl or getattr(owner, "__name__", "") == "m1triton.language":
                is_dot = True
        elif isinstance(func, ast.Name) and func.id in self.env:
            obj = self.env[func.id]
            if getattr(obj, "__module__", "") == "m1triton.language" and getattr(obj, "__name__", "") == "dot":
                is_dot = True
        if not is_dot:
            return False
        if len(call_node.args) >= 3:
            return False
        return all(kw.arg != "acc" for kw in call_node.keywords)

    def visit_Expr(self, node: ast.Expr) -> None:
        self._eval(node.value)

    def visit_Pass(self, node: ast.Pass) -> None:
        return None

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            raise NotImplementedError("GPU kernels cannot return values; write outputs with tl.store")
        self.builder.emit_return()

    def visit_Assert(self, node: ast.Assert) -> None:
        cond = self._eval(node.test)
        if isinstance(cond, _SSAVal):
            return
        msg = self._eval(node.msg) if node.msg is not None else "Kernel assertion failed"
        assert bool(cond), str(msg)

    def visit_If(self, node: ast.If) -> None:
        test_val = self._eval(node.test)
        if not isinstance(test_val, _SSAVal):
            branch = node.body if bool(test_val) else node.orelse
            for stmt in branch:
                self.visit(stmt)
            return

        cond_ssa = self._ensure_bool(test_val)
        cond_ty = self.kernel.value_type(cond_ssa.vid)
        if cond_ty.is_tile():
            raise TypeError("Runtime `if` condition must be scalar; use `tl.where` for tile predicates")

        assigned = _collect_assigned_names(list(node.body) + list(node.orelse))
        updated_vars = [name for name in assigned if name in self.env and _is_ssa_or_numeric(self.env[name])]

        pre_ssa: dict[str, _SSAVal] = {}
        result_types: list[_C.ir.Type] = []
        for name in updated_vars:
            ssa = self._materialize_ssa(self.env[name])
            pre_ssa[name] = ssa
            self.env[name] = ssa
            result_types.append(self.kernel.value_type(ssa.vid))

        with_else = bool(node.orelse) or bool(updated_vars)
        resume_block = self.builder.current_block()
        handle = self.builder.begin_if(cond_ssa.vid, result_types, with_else)

        saved_env = dict(self.env)

        self.builder.set_insertion_block(handle.then_block)
        self.env = dict(saved_env)
        for stmt in node.body:
            self.visit(stmt)
        if not self.kernel.block_has_terminator(self.builder.current_block()):
            then_yields: list[int] = []
            for idx, name in enumerate(updated_vars):
                y_ssa = self._coerce_to_match_type(self.env[name], result_types[idx])
                then_yields.append(y_ssa.vid)
            self.builder.emit_yield(then_yields)
        then_env = self.env

        else_env = dict(saved_env)
        if with_else:
            self.builder.set_insertion_block(handle.else_block)
            self.env = dict(saved_env)
            for stmt in node.orelse:
                self.visit(stmt)
            if not self.kernel.block_has_terminator(self.builder.current_block()):
                else_yields: list[int] = []
                for idx, name in enumerate(updated_vars):
                    y_ssa = self._coerce_to_match_type(self.env[name], result_types[idx])
                    else_yields.append(y_ssa.vid)
                self.builder.emit_yield(else_yields)
            else_env = self.env

        self.builder.set_insertion_block(resume_block)
        self.env = saved_env
        for idx, name in enumerate(updated_vars):
            p_then = then_env[name].ptr_params if isinstance(then_env[name], _SSAVal) else frozenset()
            p_else = else_env[name].ptr_params if isinstance(else_env[name], _SSAVal) else frozenset()
            self.env[name] = _SSAVal(vid=handle.results[idx], ptr_params=p_then | p_else)

    def _coerce_to_match_type(self, val: Any, expected_ty: _C.ir.Type) -> _SSAVal:
        if expected_ty.is_ptr_elem():
            if not isinstance(val, _SSAVal):
                raise TypeError(f"Expected pointer SSA value matching {expected_ty}, got {val!r}")
            return val
        target_dt = expected_ty.scalar_elem()
        ssa = self._coerce_to_dtype(val, target_dt)
        if expected_ty.is_tile():
            ssa = self._broadcast_to_shape(ssa, expected_ty.shape())
        return ssa

    def visit_For(self, node: ast.For) -> None:
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("`for` loop target must be a single loop variable name")
        if node.orelse:
            raise NotImplementedError("`for ... else` is not supported in M1Triton kernels")
        loop_var = node.target.id

        if not isinstance(node.iter, ast.Call):
            raise NotImplementedError("`for` loop iterator must be `range(...)` or `tl.static_range(...)`")

        iter_resolved = self._resolve_callable(node.iter.func)
        iter_kind = (iter_resolved[0], iter_resolved[1]) if iter_resolved is not None else None
        args, kwargs = self._eval_call_args(node.iter)
        if kwargs:
            raise NotImplementedError("Keyword arguments to range/static_range are not supported")

        if iter_kind == ("tl", "static_range"):
            py_bounds = [int(a) for a in args]
            for iv in range(*py_bounds):
                self.env[loop_var] = iv
                for stmt in node.body:
                    self.visit(stmt)
            return

        if iter_kind != ("builtin", "range"):
            raise NotImplementedError(f"Unsupported `for` iterator: {ast.dump(node.iter)}")

        if len(args) == 1:
            lo_raw, hi_raw, step_raw = 0, args[0], 1
        elif len(args) == 2:
            lo_raw, hi_raw, step_raw = args[0], args[1], 1
        elif len(args) == 3:
            lo_raw, hi_raw, step_raw = args[0], args[1], args[2]
        else:
            raise ValueError(f"range() expects 1 to 3 arguments, got {len(args)}")

        ir = _C.ir
        bound_dt = ir.ScalarDType.I32
        for b in (lo_raw, hi_raw, step_raw):
            if isinstance(b, _SSAVal):
                b_dt = self.kernel.value_type(b.vid).scalar_elem()
                if b_dt == ir.ScalarDType.I64:
                    bound_dt = ir.ScalarDType.I64
            elif isinstance(b, int) and not (_INT32_MIN <= b <= _INT32_MAX):
                bound_dt = ir.ScalarDType.I64

        lo_ssa = self._coerce_to_dtype(lo_raw, bound_dt)
        hi_ssa = self._coerce_to_dtype(hi_raw, bound_dt)
        step_ssa = self._coerce_to_dtype(step_raw, bound_dt)

        assigned = _collect_assigned_names(node.body)
        carried_vars = [
            name
            for name in assigned
            if name in self.env and name != loop_var and _is_ssa_or_numeric(self.env[name])
        ]

        iter_init_ssas: list[_SSAVal] = []
        iter_init_types: list[_C.ir.Type] = []
        for name in carried_vars:
            init_ssa = self._materialize_ssa(self.env[name])
            self.env[name] = init_ssa
            iter_init_ssas.append(init_ssa)
            iter_init_types.append(self.kernel.value_type(init_ssa.vid))

        resume_block = self.builder.current_block()
        handle = self.builder.begin_for_loop(
            lo_ssa.vid, hi_ssa.vid, step_ssa.vid, [s.vid for s in iter_init_ssas]
        )
        self.builder.set_insertion_block(handle.body_block)

        saved_env = dict(self.env)
        self.env = dict(saved_env)
        self.env[loop_var] = _SSAVal(handle.iv)
        self.kernel.set_value_debug_name(handle.iv, loop_var)
        for idx, name in enumerate(carried_vars):
            self.env[name] = _SSAVal(
                vid=handle.iter_args[idx],
                ptr_params=iter_init_ssas[idx].ptr_params,
                structure=iter_init_ssas[idx].structure,
            )

        for stmt in node.body:
            self.visit(stmt)

        yield_vids: list[int] = []
        yield_ptrs: list[frozenset[int]] = []
        yield_structs: list[Any] = []
        for idx, name in enumerate(carried_vars):
            y_ssa = self._coerce_to_match_type(self.env[name], iter_init_types[idx])
            yield_vids.append(y_ssa.vid)
            yield_ptrs.append(y_ssa.ptr_params)
            yield_structs.append(y_ssa.structure)

        self.builder.end_for_loop(handle, yield_vids, resume_block)
        self.builder.set_insertion_block(resume_block)

        self.env = saved_env
        for idx, name in enumerate(carried_vars):
            self.env[name] = _SSAVal(
                vid=handle.results[idx],
                ptr_params=iter_init_ssas[idx].ptr_params | yield_ptrs[idx],
                structure=yield_structs[idx],
            )

    # --- Expression visitors ---

    def _eval(self, node: ast.AST) -> Any:
        return self.visit(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.env:
            return self.env[node.id]
        if node.id in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[node.id]
        raise NameError(f"Undefined symbol in M1Triton kernel: {node.id!r}")

    def visit_Tuple(self, node: ast.Tuple) -> tuple[Any, ...]:
        return tuple(self._eval(elt) for elt in node.elts)

    def visit_List(self, node: ast.List) -> list[Any]:
        return [self._eval(elt) for elt in node.elts]

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        owner = self._eval(node.value)
        if isinstance(owner, _SSAVal):
            ty = self.kernel.value_type(owner.vid)
            if node.attr == "shape":
                return tuple(ty.shape())
            if node.attr == "dtype":
                dt = ty.ptr_elem().pointee if ty.is_ptr_elem() else ty.scalar_elem()
                return from_ir_scalar_dtype(dt)
            if node.attr == "T":
                shape = ty.shape()
                if len(shape) != 2:
                    raise ValueError(f"`.T` requires a 2D tile, got shape {shape}")
                return _SSAVal(
                    vid=self.builder.emit_permute(owner.vid, [1, 0]),
                    ptr_params=owner.ptr_params,
                )
            raise AttributeError(f"Unsupported attribute {node.attr!r} on SSA tensor value")
        return getattr(owner, node.attr)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self._eval(node.operand)
        if not isinstance(operand, _SSAVal):
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.Invert):
                return ~operand
            if isinstance(node.op, ast.Not):
                return not operand
            raise NotImplementedError(f"Unsupported unary operator: {type(node.op).__name__}")

        ir = _C.ir
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return _SSAVal(self.builder.emit_unary(ir.UnaryOp.Neg, operand.vid))
        if isinstance(node.op, (ast.Invert, ast.Not)):
            return _SSAVal(self.builder.emit_unary(ir.UnaryOp.Not, operand.vid))
        raise NotImplementedError(f"Unsupported SSA unary operator: {type(node.op).__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        lhs = self._eval(node.left)
        rhs = self._eval(node.right)
        return self._apply_binop(node.op, lhs, rhs)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        ir = _C.ir
        if isinstance(node.op, ast.And):
            acc: Any = None
            for idx, v_node in enumerate(node.values):
                cur = self._eval(v_node)
                if idx == 0:
                    acc = cur
                    if not isinstance(acc, _SSAVal) and not bool(acc):
                        return acc
                elif not isinstance(acc, _SSAVal) and not isinstance(cur, _SSAVal):
                    acc = acc and cur
                    if not bool(acc):
                        return acc
                else:
                    lhs_b = self._ensure_bool(acc)
                    rhs_b = self._ensure_bool(cur)
                    lhs_b, rhs_b, _ = self._align_binary_operands(lhs_b, rhs_b)
                    acc = _SSAVal(self.builder.emit_binary(ir.BinaryOp.And, lhs_b.vid, rhs_b.vid))
            return acc
        if isinstance(node.op, ast.Or):
            acc = None
            for idx, v_node in enumerate(node.values):
                cur = self._eval(v_node)
                if idx == 0:
                    acc = cur
                    if not isinstance(acc, _SSAVal) and bool(acc):
                        return acc
                elif not isinstance(acc, _SSAVal) and not isinstance(cur, _SSAVal):
                    acc = acc or cur
                    if bool(acc):
                        return acc
                else:
                    lhs_b = self._ensure_bool(acc)
                    rhs_b = self._ensure_bool(cur)
                    lhs_b, rhs_b, _ = self._align_binary_operands(lhs_b, rhs_b)
                    acc = _SSAVal(self.builder.emit_binary(ir.BinaryOp.Or, lhs_b.vid, rhs_b.vid))
            return acc
        raise NotImplementedError(f"Unsupported boolean op: {type(node.op).__name__}")

    def visit_Compare(self, node: ast.Compare) -> Any:
        ir = _C.ir
        cmp_map = {
            ast.Eq: ir.BinaryOp.CmpEq,
            ast.NotEq: ir.BinaryOp.CmpNe,
            ast.Lt: ir.BinaryOp.CmpLt,
            ast.LtE: ir.BinaryOp.CmpLe,
            ast.Gt: ir.BinaryOp.CmpGt,
            ast.GtE: ir.BinaryOp.CmpGe,
        }
        left = self._eval(node.left)
        acc: Any = None
        for op_node, comp_node in zip(node.ops, node.comparators):
            right = self._eval(comp_node)
            if not isinstance(left, _SSAVal) and not isinstance(right, _SSAVal):
                py_fn = _PY_CMPOPS.get(type(op_node))
                if py_fn is None:
                    raise NotImplementedError(f"Unsupported comparison: {type(op_node).__name__}")
                part: Any = py_fn(left, right)
            else:
                ir_op = cmp_map.get(type(op_node))
                if ir_op is None:
                    raise NotImplementedError(f"Unsupported SSA comparison: {type(op_node).__name__}")
                lhs_ssa, rhs_ssa, _ = self._align_binary_operands(left, right)
                part = _SSAVal(self.builder.emit_binary(ir_op, lhs_ssa.vid, rhs_ssa.vid))

            if acc is None:
                acc = part
            elif not isinstance(acc, _SSAVal) and not isinstance(part, _SSAVal):
                acc = bool(acc) and bool(part)
            else:
                a_b = self._ensure_bool(acc)
                p_b = self._ensure_bool(part)
                a_b, p_b, _ = self._align_binary_operands(a_b, p_b)
                acc = _SSAVal(self.builder.emit_binary(ir.BinaryOp.And, a_b.vid, p_b.vid))
            left = right
        return acc

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        cond = self._eval(node.test)
        if not isinstance(cond, _SSAVal):
            return self._eval(node.body) if bool(cond) else self._eval(node.orelse)
        t_val = self._eval(node.body)
        f_val = self._eval(node.orelse)
        return self._lower_select(cond, t_val, f_val)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        base = self._eval(node.value)
        if not isinstance(base, _SSAVal):
            idx = self._eval_slice_or_expr(node.slice)
            return base[idx]

        slice_node = node.slice
        elts = list(slice_node.elts) if isinstance(slice_node, ast.Tuple) else [slice_node]
        cur = base
        out_axis = 0
        for elt in elts:
            if isinstance(elt, ast.Constant) and elt.value is None:
                cur = _SSAVal(
                    vid=self.builder.emit_expand_dims(cur.vid, out_axis),
                    ptr_params=cur.ptr_params,
                    structure=cur.structure,
                )
                out_axis += 1
            elif isinstance(elt, ast.Slice) and elt.lower is None and elt.upper is None and elt.step is None:
                out_axis += 1
            else:
                raise NotImplementedError(
                    f"Only `:` and `None` expand_dims indexing are supported on SSA tiles, got {ast.dump(elt)}"
                )
        return cur

    def _eval_slice_or_expr(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Slice):
            lo = self._eval(node.lower) if node.lower is not None else None
            hi = self._eval(node.upper) if node.upper is not None else None
            step = self._eval(node.step) if node.step is not None else None
            return slice(lo, hi, step)
        return self._eval(node)

    # --- Call dispatch (`tl.*`, `tlx.*`, SSA methods, Python builtins) ---

    def _eval_call_args(self, call_node: ast.Call) -> tuple[list[Any], dict[str, Any]]:
        args = [self._eval(a) for a in call_node.args]
        kwargs = {kw.arg: self._eval(kw.value) for kw in call_node.keywords if kw.arg is not None}
        return args, kwargs

    def _resolve_callable(self, func_node: ast.AST) -> tuple[str, str, Any] | None:
        if isinstance(func_node, ast.Attribute):
            owner = self._eval(func_node.value)
            if isinstance(owner, _SSAVal):
                return ("ssa_method", func_node.attr, owner)
            mod_name = getattr(owner, "__name__", "")
            if owner is tl or mod_name == "m1triton.language":
                return ("tl", func_node.attr, owner)
            if owner is tlx or mod_name == "m1triton.tlx":
                return ("tlx", func_node.attr, owner)
            if mod_name in ("m1triton", "m1triton.triton_compat") and func_node.attr == "cdiv":
                return ("tl", "cdiv", owner)
            return ("py_attr", func_node.attr, owner)

        if isinstance(func_node, ast.Name):
            if func_node.id in ("range", "min", "max", "abs", "int", "float", "bool", "len", "tuple", "list"):
                if func_node.id not in self.env:
                    return ("builtin", func_node.id, None)
            if func_node.id in self.env:
                obj = self.env[func_node.id]
                obj_mod = getattr(obj, "__module__", "")
                if obj_mod == "m1triton.language":
                    return ("tl", getattr(obj, "__name__", func_node.id), obj)
                if obj_mod == "m1triton.tlx":
                    return ("tlx", getattr(obj, "__name__", func_node.id), obj)
        return None

    def visit_Call(self, node: ast.Call) -> Any:
        resolved = self._resolve_callable(node.func)
        args, kwargs = self._eval_call_args(node)

        if resolved is not None:
            ns, name, owner = resolved
            if ns == "ssa_method":
                return self._lower_ssa_method(owner, name, args, kwargs)
            if ns == "tl":
                return self._lower_tl_call(name, args, kwargs)
            if ns == "tlx":
                return self._lower_tlx_call(name, args, kwargs)
            if ns == "builtin":
                return self._lower_builtin_call(name, args, kwargs)
            if ns == "py_attr":
                fn_obj = getattr(owner, name)
                if any(isinstance(a, _SSAVal) for a in args) or any(isinstance(v, _SSAVal) for v in kwargs.values()):
                    raise NotImplementedError(f"Unsupported runtime call on SSA value: {fn_obj!r}")
                return fn_obj(*args, **kwargs)

        fn_obj = self._eval(node.func)
        if any(isinstance(a, _SSAVal) for a in args) or any(isinstance(v, _SSAVal) for v in kwargs.values()):
            raise NotImplementedError(f"Unsupported runtime call on SSA value: {fn_obj!r}")
        return fn_obj(*args, **kwargs)

    def _lower_ssa_method(self, owner: _SSAVal, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if name == "to":
            dt = args[0] if args else kwargs["dtype"]
            return self._coerce_to_dtype(owner, to_ir_scalar_dtype(dt))
        raise AttributeError(f"Unsupported method `.{name}()` on SSA value")

    def _lower_builtin_call(self, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        ir = _C.ir
        has_ssa = any(isinstance(a, _SSAVal) for a in args)
        if not has_ssa:
            builtins_map = {
                "min": min,
                "max": max,
                "abs": abs,
                "int": int,
                "float": float,
                "bool": bool,
                "len": len,
                "tuple": tuple,
                "list": list,
                "range": range,
            }
            return builtins_map[name](*args, **kwargs)
        if name == "min" and len(args) == 2:
            return self._lower_tl_binary(ir.BinaryOp.Min, args[0], args[1])
        if name == "max" and len(args) == 2:
            return self._lower_tl_binary(ir.BinaryOp.Max, args[0], args[1])
        if name == "abs" and len(args) == 1:
            return _SSAVal(self.builder.emit_unary(ir.UnaryOp.Abs, args[0].vid))
        if name == "float" and len(args) == 1:
            return self._coerce_to_dtype(args[0], ir.ScalarDType.F32)
        if name == "int" and len(args) == 1:
            return self._coerce_to_dtype(args[0], ir.ScalarDType.I32)
        if name == "bool" and len(args) == 1:
            return self._ensure_bool(args[0])
        raise NotImplementedError(f"Unsupported builtin `{name}` on SSA arguments")

    def _lower_select(self, cond: Any, true_val: Any, false_val: Any) -> _SSAVal:
        cond_ssa = self._ensure_bool(cond)
        t_is_ptr = isinstance(true_val, _SSAVal) and self.kernel.value_type(true_val.vid).is_ptr_elem()
        f_is_ptr = isinstance(false_val, _SSAVal) and self.kernel.value_type(false_val.vid).is_ptr_elem()
        if t_is_ptr or f_is_ptr:
            if not (t_is_ptr and f_is_ptr):
                raise TypeError("tl.where cannot mix pointer and non-pointer branches")
            t_ty = self.kernel.value_type(true_val.vid)
            f_ty = self.kernel.value_type(false_val.vid)
            if t_ty.ptr_elem() != f_ty.ptr_elem():
                raise TypeError("tl.where pointer branches must have matching pointee types")
            target_shape = _broadcast_shapes(
                self.kernel.value_type(cond_ssa.vid).shape(),
                _broadcast_shapes(t_ty.shape(), f_ty.shape()),
            )
            t_ssa, f_ssa = true_val, false_val
            if target_shape:
                cond_ssa = self._broadcast_to_shape(cond_ssa, target_shape)
                t_ssa = self._broadcast_to_shape(t_ssa, target_shape)
                f_ssa = self._broadcast_to_shape(f_ssa, target_shape)
            vid = self.builder.emit_select(cond_ssa.vid, t_ssa.vid, f_ssa.vid)
            return _SSAVal(vid=vid, ptr_params=t_ssa.ptr_params | f_ssa.ptr_params)

        t_ssa, f_ssa, _ = self._align_binary_operands(true_val, false_val)
        c_shape = self.kernel.value_type(cond_ssa.vid).shape()
        v_shape = self.kernel.value_type(t_ssa.vid).shape()
        target_shape = _broadcast_shapes(c_shape, v_shape)
        if target_shape:
            cond_ssa = self._broadcast_to_shape(cond_ssa, target_shape)
            t_ssa = self._broadcast_to_shape(t_ssa, target_shape)
            f_ssa = self._broadcast_to_shape(f_ssa, target_shape)
        vid = self.builder.emit_select(cond_ssa.vid, t_ssa.vid, f_ssa.vid)
        return _SSAVal(vid=vid, ptr_params=t_ssa.ptr_params | f_ssa.ptr_params)

    def _lower_tl_binary(self, op: _C.ir.BinaryOp, lhs: Any, rhs: Any) -> _SSAVal:
        lhs_ssa, rhs_ssa, _ = self._align_binary_operands(lhs, rhs)
        return _SSAVal(self.builder.emit_binary(op, lhs_ssa.vid, rhs_ssa.vid))

    def _lower_tl_dot(self, args: list[Any], kwargs: dict[str, Any]) -> _SSAVal:
        ir = _C.ir
        lhs = args[0] if len(args) > 0 else kwargs["input"]
        rhs = args[1] if len(args) > 1 else kwargs["other"]
        acc = args[2] if len(args) > 2 else kwargs.get("acc")
        allow_tf32 = bool(_unwrap_const(kwargs.get("allow_tf32", False)))
        out_dtype = _unwrap_const(kwargs.get("out_dtype", tl.float32))
        trans_a = bool(_unwrap_const(kwargs.get("trans_a", kwargs.get("transpose_a", False))))
        trans_b = bool(_unwrap_const(kwargs.get("trans_b", kwargs.get("transpose_b", False))))

        if not isinstance(lhs, _SSAVal) or not isinstance(rhs, _SSAVal):
            raise TypeError("tl.dot requires SSA tile operands")

        attr = ir.DotAttr()
        attr.allow_tf32 = allow_tf32
        attr.transpose_a = trans_a
        attr.transpose_b = trans_b
        if acc is not None:
            acc_ssa = self._materialize_ssa(acc)
            attr.acc_dtype = self.kernel.value_type(acc_ssa.vid).scalar_elem()
            acc_vid = acc_ssa.vid
        else:
            attr.acc_dtype = to_ir_scalar_dtype(out_dtype)
            acc_vid = ir.INVALID_ID

        kernel_struct = self.kernel.hints.structure
        lhs_ty = self.kernel.value_type(lhs.vid)
        rhs_ty = self.kernel.value_type(rhs.vid)
        lhs_ty_struct = lhs_ty.as_tile().structure if lhs_ty.is_tile() else ir.StructureAttr()
        rhs_ty_struct = rhs_ty.as_tile().structure if rhs_ty.is_tile() else ir.StructureAttr()

        if lhs.structure is not None:
            attr.lhs_struct = lhs.structure
        elif lhs_ty_struct.kind != ir.MatrixStructure.Dense:
            attr.lhs_struct = lhs_ty_struct
        elif kernel_struct is not None:
            attr.lhs_struct = kernel_struct

        if rhs.structure is not None:
            attr.rhs_struct = rhs.structure
        elif rhs_ty_struct.kind != ir.MatrixStructure.Dense:
            attr.rhs_struct = rhs_ty_struct
        elif kernel_struct is not None:
            attr.rhs_struct = kernel_struct

        closed_out_kinds = (
            ir.MatrixStructure.UpperTriangular,
            ir.MatrixStructure.LowerTriangular,
            ir.MatrixStructure.Diagonal,
            ir.MatrixStructure.CausalMask,
        )
        if kernel_struct is not None and kernel_struct.kind in closed_out_kinds:
            attr.out_struct = kernel_struct
        elif (
            attr.lhs_struct.kind == attr.rhs_struct.kind
            and attr.lhs_struct.kind in closed_out_kinds
            and attr.lhs_struct.diagonal_offset == 0
            and attr.rhs_struct.diagonal_offset == 0
        ):
            attr.out_struct = ir.StructureAttr(attr.lhs_struct.kind, 0, False)

        hints = ir.OpHints()
        if self.kernel.hints.tile_spec is not None:
            hints.tile_spec = self.kernel.hints.tile_spec
        if kernel_struct is not None:
            hints.structure = kernel_struct
        elif attr.out_struct.kind != ir.MatrixStructure.Dense:
            hints.structure = attr.out_struct
        elif attr.lhs_struct.kind != ir.MatrixStructure.Dense:
            hints.structure = attr.lhs_struct
        elif attr.rhs_struct.kind != ir.MatrixStructure.Dense:
            hints.structure = attr.rhs_struct
        if self.kernel.hints.persistent_grid is not None:
            hints.persistent_grid = self.kernel.hints.persistent_grid

        vid = self.builder.emit_dot(lhs.vid, rhs.vid, acc_vid, attr, hints)
        out_struct = attr.out_struct if attr.out_struct.kind != ir.MatrixStructure.Dense else None
        return _SSAVal(vid=vid, structure=out_struct)

    def _lower_tl_reduce(self, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        ir = _C.ir
        x = args[0] if len(args) > 0 else kwargs["input"]
        axis = _unwrap_const(args[1] if len(args) > 1 else kwargs.get("axis"))
        keep_dims = bool(
            _unwrap_const(
                args[2] if len(args) > 2 else kwargs.get("keep_dims", kwargs.get("keepdims", False))
            )
        )
        return_indices = bool(
            _unwrap_const(args[3] if len(args) > 3 else kwargs.get("return_indices", False))
        )

        x_ssa = self._materialize_ssa(x)
        x_ty = self.kernel.value_type(x_ssa.vid)
        if not x_ty.is_tile() or x_ty.is_ptr_elem():
            raise TypeError(f"tl.{name} requires a non-pointer tile operand, got {x_ty}")

        shape = list(x_ty.shape())
        rank = len(shape)
        if rank == 0:
            raise ValueError(f"tl.{name} cannot reduce a 0-D tile")

        kind_map = {
            "sum": ir.ReduceKind.Sum,
            "max": ir.ReduceKind.Max,
            "min": ir.ReduceKind.Min,
            "argmax": ir.ReduceKind.Argmax,
            "argmin": ir.ReduceKind.Argmin,
        }
        kind = kind_map[name]
        if name in ("argmax", "argmin"):
            result_mode = ir.ReduceResultMode.IndexOnly
        elif return_indices:
            result_mode = ir.ReduceResultMode.ValueAndIndex
        else:
            result_mode = ir.ReduceResultMode.ValueOnly

        if axis is None:
            if rank == 1:
                axes = [0]
            elif name in ("sum", "max", "min") and not return_indices:
                axes = list(range(rank - 1, -1, -1))
            else:
                raise ValueError(f"tl.{name} requires an explicit axis for tiles of rank {rank}")
        else:
            ax = int(axis)
            if ax < 0:
                ax += rank
            if ax < 0 or ax >= rank:
                raise ValueError(f"tl.{name} axis {axis} out of range for tile rank {rank}")
            axes = [ax]

        cur = x_ssa
        for ax in axes:
            attr = ir.ReduceAttr()
            attr.kind = kind
            attr.axis = ax
            attr.keep_dims = keep_dims
            attr.result_mode = result_mode
            vid = self.builder.emit_reduce(cur.vid, attr)
            if result_mode == ir.ReduceResultMode.ValueAndIndex:
                idx_vid = vid + 1
                if self.kernel.value_def_op(idx_vid) != self.kernel.value_def_op(vid):
                    raise RuntimeError("Unexpected Reduce ValueAndIndex result ValueId layout")
                return (_SSAVal(vid), _SSAVal(idx_vid))
            cur = _SSAVal(vid)
        return cur

    def _lower_tl_call(self, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        ir = _C.ir

        if name == "program_id":
            axis = int(args[0] if args else kwargs["axis"])
            return _SSAVal(self.builder.emit_program_id(axis))

        if name == "num_programs":
            axis = int(args[0] if args else kwargs["axis"])
            return _SSAVal(self.builder.emit_num_programs(axis))

        if name == "arange":
            start = int(args[0] if len(args) > 0 else kwargs["start"])
            end = int(args[1] if len(args) > 1 else kwargs["end"])
            return _SSAVal(self.builder.emit_arange(start, end))

        if name == "zeros":
            shape = args[0] if len(args) > 0 else kwargs["shape"]
            if isinstance(shape, int):
                shape = (shape,)
            dt = args[1] if len(args) > 1 else kwargs.get("dtype", tl.float32)
            ir_dt = to_ir_scalar_dtype(dt)
            zero_val = self._emit_const(0, ir_dt)
            return _SSAVal(self.builder.emit_splat(zero_val.vid, [int(d) for d in shape]))

        if name == "full":
            shape = args[0] if len(args) > 0 else kwargs["shape"]
            if isinstance(shape, int):
                shape = (shape,)
            fill_v = args[1] if len(args) > 1 else kwargs["value"]
            dt = args[2] if len(args) > 2 else kwargs.get("dtype")
            if dt is not None:
                ir_dt = to_ir_scalar_dtype(dt)
                c_val = self._coerce_to_dtype(fill_v, ir_dt)
            else:
                c_val = self._materialize_ssa(fill_v)
            return _SSAVal(self.builder.emit_splat(c_val.vid, [int(d) for d in shape]))

        if name == "load":
            ptrs = args[0] if len(args) > 0 else kwargs["pointer"]
            mask = args[1] if len(args) > 1 else kwargs.get("mask")
            other = args[2] if len(args) > 2 else kwargs.get("other")
            volatile = bool(kwargs.get("volatile", False))

            if not isinstance(ptrs, _SSAVal):
                raise TypeError("tl.load pointer argument must be an SSA pointer or tile of pointers")
            ptrs_ty = self.kernel.value_type(ptrs.vid)
            pointee_dt = ptrs_ty.ptr_elem().pointee
            target_shape = list(ptrs_ty.shape())

            mask_vid = ir.INVALID_ID
            if mask is not None:
                mask_ssa = self._ensure_bool(mask)
                if target_shape:
                    mask_ssa = self._broadcast_to_shape(mask_ssa, target_shape)
                mask_vid = mask_ssa.vid

            other_vid = ir.INVALID_ID
            if other is not None:
                other_ssa = self._coerce_to_dtype(other, pointee_dt)
                if target_shape:
                    other_ssa = self._broadcast_to_shape(other_ssa, target_shape)
                other_vid = other_ssa.vid

            attr = ir.LoadStoreAttr()
            attr.alignment_bytes = _scalar_dtype_bytes(pointee_dt)
            attr.vec_width = 1
            attr.has_mask = mask_vid != ir.INVALID_ID
            attr.has_other = other_vid != ir.INVALID_ID
            attr.is_volatile = volatile
            return _SSAVal(self.builder.emit_load(ptrs.vid, mask_vid, other_vid, attr))

        if name == "store":
            ptrs = args[0] if len(args) > 0 else kwargs["pointer"]
            value = args[1] if len(args) > 1 else kwargs["value"]
            mask = args[2] if len(args) > 2 else kwargs.get("mask")

            if not isinstance(ptrs, _SSAVal):
                raise TypeError("tl.store pointer argument must be an SSA pointer or tile of pointers")
            for pidx in ptrs.ptr_params:
                self.kernel.set_param_writable(pidx, True)

            ptrs_ty = self.kernel.value_type(ptrs.vid)
            pointee_dt = ptrs_ty.ptr_elem().pointee
            target_shape = list(ptrs_ty.shape())

            val_ssa = self._coerce_to_dtype(value, pointee_dt)
            if target_shape:
                val_ssa = self._broadcast_to_shape(val_ssa, target_shape)

            mask_vid = ir.INVALID_ID
            if mask is not None:
                mask_ssa = self._ensure_bool(mask)
                if target_shape:
                    mask_ssa = self._broadcast_to_shape(mask_ssa, target_shape)
                mask_vid = mask_ssa.vid

            attr = ir.LoadStoreAttr()
            attr.alignment_bytes = _scalar_dtype_bytes(pointee_dt)
            attr.vec_width = 1
            attr.has_mask = mask_vid != ir.INVALID_ID
            self.builder.emit_store(ptrs.vid, val_ssa.vid, mask_vid, attr)
            return None

        if name == "atomic_add":
            ptrs = args[0] if len(args) > 0 else kwargs["pointer"]
            value = args[1] if len(args) > 1 else kwargs["val"]
            mask = args[2] if len(args) > 2 else kwargs.get("mask")

            if not isinstance(ptrs, _SSAVal):
                raise TypeError("tl.atomic_add pointer argument must be an SSA pointer or tile of pointers")
            for pidx in ptrs.ptr_params:
                self.kernel.set_param_writable(pidx, True)

            ptrs_ty = self.kernel.value_type(ptrs.vid)
            pointee_dt = ptrs_ty.ptr_elem().pointee
            target_shape = list(ptrs_ty.shape())

            val_ssa = self._coerce_to_dtype(value, pointee_dt)
            if target_shape:
                val_ssa = self._broadcast_to_shape(val_ssa, target_shape)

            mask_vid = ir.INVALID_ID
            if mask is not None:
                mask_ssa = self._ensure_bool(mask)
                if target_shape:
                    mask_ssa = self._broadcast_to_shape(mask_ssa, target_shape)
                mask_vid = mask_ssa.vid
            return _SSAVal(self.builder.emit_atomic_add(ptrs.vid, val_ssa.vid, mask_vid))

        if name == "dot":
            return self._lower_tl_dot(args, kwargs)

        if name in ("sum", "max", "min", "argmax", "argmin"):
            return self._lower_tl_reduce(name, args, kwargs)

        if name in ("cumsum", "cumprod"):
            x = args[0] if len(args) > 0 else kwargs["input"]
            axis = int(args[1] if len(args) > 1 else kwargs.get("axis", 0))
            reverse = bool(args[2] if len(args) > 2 else kwargs.get("reverse", False))
            attr = ir.ScanAttr()
            attr.kind = ir.ScanKind.Cumsum if name == "cumsum" else ir.ScanKind.Cumprod
            attr.axis = axis
            attr.exclusive = False
            attr.reverse = reverse
            attr.has_mask = False
            x_ssa = self._materialize_ssa(x)
            return _SSAVal(self.builder.emit_scan(x_ssa.vid, attr, ir.INVALID_ID))

        if name == "where":
            cond = args[0] if len(args) > 0 else kwargs["condition"]
            x = args[1] if len(args) > 1 else kwargs["x"]
            y = args[2] if len(args) > 2 else kwargs["y"]
            return self._lower_select(cond, x, y)

        if name == "cast":
            x = args[0] if len(args) > 0 else kwargs["input"]
            dt = args[1] if len(args) > 1 else kwargs["dtype"]
            return self._coerce_to_dtype(x, to_ir_scalar_dtype(dt))

        if name == "trans":
            x = self._materialize_ssa(args[0] if len(args) > 0 else kwargs["input"])
            dims = list(args[1:]) if len(args) > 1 else None
            if dims is None:
                rank = len(self.kernel.value_type(x.vid).shape())
                dims = list(reversed(range(rank)))
            return _SSAVal(self.builder.emit_permute(x.vid, [int(d) for d in dims]), ptr_params=x.ptr_params)

        if name == "reshape":
            x = self._materialize_ssa(args[0] if len(args) > 0 else kwargs["input"])
            shape = args[1] if len(args) > 1 else kwargs["shape"]
            return _SSAVal(self.builder.emit_reshape(x.vid, [int(d) for d in shape]), ptr_params=x.ptr_params)

        if name == "broadcast_to":
            x = self._materialize_ssa(args[0] if len(args) > 0 else kwargs["input"])
            shape = args[1] if len(args) > 1 else kwargs["shape"]
            return self._broadcast_to_shape(x, shape)

        if name == "expand_dims":
            x = self._materialize_ssa(args[0] if len(args) > 0 else kwargs["input"])
            axis = int(args[1] if len(args) > 1 else kwargs["axis"])
            return _SSAVal(self.builder.emit_expand_dims(x.vid, axis), ptr_params=x.ptr_params)

        unary_map = {
            "exp": ir.UnaryOp.Exp,
            "log": ir.UnaryOp.Log,
            "sqrt": ir.UnaryOp.Sqrt,
            "rsqrt": ir.UnaryOp.Rsqrt,
            "abs": ir.UnaryOp.Abs,
            "sigmoid": ir.UnaryOp.Sigmoid,
            "tanh": ir.UnaryOp.Tanh,
            "erf": ir.UnaryOp.Erf,
            "floor": ir.UnaryOp.Floor,
            "ceil": ir.UnaryOp.Ceil,
            "sin": ir.UnaryOp.Sin,
            "cos": ir.UnaryOp.Cos,
        }
        if name in unary_map:
            x = args[0] if args else kwargs["x"]
            x_ssa = self._materialize_ssa(x)
            if name in _FLOAT_ONLY_UNARY_OPS:
                cur_dt = self.kernel.value_type(x_ssa.vid).scalar_elem()
                if not _is_float_dtype(cur_dt):
                    x_ssa = self._coerce_to_dtype(x_ssa, ir.ScalarDType.F32)
            return _SSAVal(self.builder.emit_unary(unary_map[name], x_ssa.vid))

        if name == "maximum":
            x = args[0] if len(args) > 0 else kwargs["x"]
            y = args[1] if len(args) > 1 else kwargs["y"]
            if not isinstance(x, _SSAVal) and not isinstance(y, _SSAVal):
                return max(x, y)
            return self._lower_tl_binary(ir.BinaryOp.Max, x, y)

        if name == "minimum":
            x = args[0] if len(args) > 0 else kwargs["x"]
            y = args[1] if len(args) > 1 else kwargs["y"]
            if not isinstance(x, _SSAVal) and not isinstance(y, _SSAVal):
                return min(x, y)
            return self._lower_tl_binary(ir.BinaryOp.Min, x, y)

        if name == "cdiv":
            x = args[0] if len(args) > 0 else kwargs["x"]
            y = args[1] if len(args) > 1 else kwargs["y"]
            if not isinstance(x, _SSAVal) and not isinstance(y, _SSAVal):
                return (int(x) + int(y) - 1) // int(y)
            num = self._apply_binop(ast.Sub(), self._apply_binop(ast.Add(), x, y), 1)
            return self._apply_binop(ast.FloorDiv(), num, y)

        if name in ("multiple_of", "max_contiguous"):
            return args[0] if args else kwargs["input"]

        if name == "static_assert":
            cond = args[0] if args else kwargs["cond"]
            msg = args[1] if len(args) > 1 else kwargs.get("msg", "")
            assert bool(cond), str(msg or "tl.static_assert failed")
            return None

        raise NotImplementedError(f"Unsupported tl.{name} primitive in M1 AST lowering")

    def _lower_tlx_call(self, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        ir = _C.ir

        if name == "vec_width":
            width = int(args[0] if args else kwargs["width"])
            hint = tlx.VecWidthHint(width=width)
            _apply_kernel_hint(self.kernel, hint)
            return hint

        if name == "tile_spec":
            u_args = [_unwrap_const(a) for a in args]
            u_kwargs = {k: _unwrap_const(v) for k, v in kwargs.items()}
            hint = tlx.tile_spec(*u_args, **u_kwargs)
            _apply_kernel_hint(self.kernel, hint)
            return hint

        if name == "layout_hint":
            u_args = [_unwrap_const(a) for a in args]
            u_kwargs = {k: _unwrap_const(v) for k, v in kwargs.items()}
            hint = tlx.layout_hint(*u_args, **u_kwargs)
            _apply_kernel_hint(self.kernel, hint)
            return hint

        if name == "persistent_grid":
            u_args = [_unwrap_const(a) for a in args]
            u_kwargs = {k: _unwrap_const(v) for k, v in kwargs.items()}
            hint = tlx.persistent_grid(*u_args, **u_kwargs)
            _apply_kernel_hint(self.kernel, hint)
            return hint

        if name == "epilogue":
            raw_steps = _unwrap_const(args[0] if args else kwargs["steps"])
            if not raw_steps:
                return None
            param_idx_by_name = {kp.name: idx for idx, kp in enumerate(self.kernel.params)}
            param_idx_by_vid = {self.param_vals[kp.name].vid: idx for idx, kp in enumerate(self.kernel.params) if kp.name in self.param_vals}
            step_hints: list[tlx.EpilogueStepHint] = []
            for raw in raw_steps:
                if isinstance(raw, tlx.EpilogueStepHint):
                    step_hints.append(raw)
                    continue
                if isinstance(raw, str):
                    op_key = raw.strip().lower().replace("-", "_")
                    if op_key == "softplus":
                        step_hints.append(tlx.EpilogueStepHint(op=raw, param0=1.0, param1=20.0))
                    else:
                        step_hints.append(tlx.EpilogueStepHint(op=raw))
                    continue
                if not isinstance(raw, (tuple, list)) or len(raw) == 0:
                    raise TypeError(f"Invalid tlx.epilogue step: {raw!r}")
                op_name = str(_unwrap_const(raw[0]))
                op_key = op_name.strip().lower().replace("-", "_")
                p0, p1, aidx = 0.0, 0.0, -1
                if op_key in {
                    "add_bias", "sub_bias", "mul_bias", "div_bias", "min_bias", "max_bias",
                    "add_scalar", "sub_scalar", "mul_scalar", "div_scalar",
                    "add_residual", "sub_residual", "mul_residual",
                }:
                    if len(raw) < 2:
                        raise ValueError(f"tlx.epilogue {op_name} requires a kernel parameter operand")
                    ref = _unwrap_const(raw[1])
                    if isinstance(ref, _SSAVal):
                        if ref.vid not in param_idx_by_vid:
                            raise ValueError(f"tlx.epilogue {op_name} operand must be a direct kernel parameter")
                        aidx = param_idx_by_vid[ref.vid]
                    elif isinstance(ref, str):
                        if ref not in param_idx_by_name:
                            raise ValueError(f"tlx.epilogue unknown kernel param {ref!r}")
                        aidx = param_idx_by_name[ref]
                    else:
                        aidx = int(ref)
                elif op_key in {"save_reg", "add_saved", "sub_saved", "mul_saved"}:
                    if len(raw) < 2:
                        raise ValueError(f"tlx.epilogue {op_name} requires a register slot index")
                    aidx = int(_unwrap_const(raw[1]))
                elif op_key == "softplus":
                    p0 = float(_unwrap_const(raw[1])) if len(raw) >= 2 and _unwrap_const(raw[1]) is not None else 1.0
                    p1 = float(_unwrap_const(raw[2])) if len(raw) >= 3 and _unwrap_const(raw[2]) is not None else 20.0
                else:
                    if len(raw) >= 2 and _unwrap_const(raw[1]) is not None:
                        p0 = float(_unwrap_const(raw[1]))
                    if len(raw) >= 3 and _unwrap_const(raw[2]) is not None:
                        p1 = float(_unwrap_const(raw[2]))
                    if len(raw) >= 4 and _unwrap_const(raw[3]) is not None:
                        aidx = int(_unwrap_const(raw[3]))
                step_hints.append(tlx.EpilogueStepHint(op=op_name, param0=p0, param1=p1, arg_idx=aidx))
            hint = tlx.EpilogueDAGHint(steps=tuple(step_hints))
            _apply_kernel_hint(self.kernel, hint)
            return hint

        if name == "fused_tail":
            u_args = [_unwrap_const(a) for a in args]
            u_kwargs = {k: _unwrap_const(v) for k, v in kwargs.items()}
            if not u_args or u_args[0] is None or str(u_args[0]).lower() == "none":
                return None
            hint = tlx.fused_tail(*u_args, **u_kwargs)
            _apply_kernel_hint(self.kernel, hint)
            return hint

        if name == "structure":
            diag_off = int(_unwrap_const(kwargs.get("diagonal_offset", 0)))
            unit_diag = bool(_unwrap_const(kwargs.get("is_unit_diagonal", False)))
            if args and isinstance(args[0], _SSAVal):
                val = args[0]
                kind = str(_unwrap_const(args[1] if len(args) > 1 else kwargs["kind"]))
                s_attr = _parse_structure_attr(kind, diag_off, unit_diag)
                self.kernel.set_value_structure(val.vid, s_attr)
                def_op = self.kernel.value_def_op(val.vid)
                if def_op != ir.INVALID_ID:
                    oh = ir.OpHints()
                    oh.structure = s_attr
                    self.kernel.set_op_hints(def_op, oh)
                return _SSAVal(vid=val.vid, ptr_params=val.ptr_params, structure=s_attr)
            raw_kind = _unwrap_const(args[0] if args else kwargs["kind"])
            if raw_kind is None:
                return None
            kind = str(raw_kind)
            hint = tlx.StructureHint(kind=kind, diagonal_offset=diag_off, is_unit_diagonal=unit_diag)
            _apply_kernel_hint(self.kernel, hint)
            return hint

        if name == "online_softmax":
            x_ptr = args[0] if len(args) > 0 else kwargs["x_ptr"]
            out_ptr = args[1] if len(args) > 1 else kwargs["out_ptr"]
            num_rows = self._materialize_ssa(args[2] if len(args) > 2 else kwargs["num_rows"])
            row_len = self._materialize_ssa(args[3] if len(args) > 3 else kwargs["row_len"])
            if not isinstance(x_ptr, _SSAVal) or not isinstance(out_ptr, _SSAVal):
                raise TypeError("tlx.online_softmax expects SSA pointer operands for x_ptr and out_ptr")
            for pidx in out_ptr.ptr_params:
                self.kernel.set_param_writable(pidx, True)
            attr = ir.OnlineSoftmaxAttr()
            attr.axis = int(_unwrap_const(kwargs.get("axis", -1)))
            attr.log_softmax = bool(_unwrap_const(kwargs.get("log_softmax", False)))
            self.builder.emit_online_softmax(x_ptr.vid, out_ptr.vid, num_rows.vid, row_len.vid, attr, self.kernel.hints)
            return None

        if name == "welford_norm":
            kind_str = str(_unwrap_const(args[0] if len(args) > 0 else kwargs["kind"])).strip().lower()
            raw_ops = list(args[1:]) if len(args) > 1 else list(kwargs.get("operands", ()))
            if len(raw_ops) < 3:
                raise ValueError("tlx.welford_norm requires at least (x_ptr, out_ptr, dim0)")
            ssa_ops = [self._materialize_ssa(v) for v in raw_ops]
            for pidx in ssa_ops[1].ptr_params:
                self.kernel.set_param_writable(pidx, True)
            kind_map = {
                "layer_norm": ir.NormKind.LayerNorm,
                "layernorm": ir.NormKind.LayerNorm,
                "batch_norm": ir.NormKind.BatchNorm,
                "batchnorm": ir.NormKind.BatchNorm,
                "instance_norm": ir.NormKind.InstanceNorm,
                "instancenorm": ir.NormKind.InstanceNorm,
                "group_norm": ir.NormKind.GroupNorm,
                "groupnorm": ir.NormKind.GroupNorm,
                "rms_norm": ir.NormKind.RMSNorm,
                "rmsnorm": ir.NormKind.RMSNorm,
                "frobenius_norm": ir.NormKind.FrobeniusNorm,
                "frobenius": ir.NormKind.FrobeniusNorm,
                "l1_norm": ir.NormKind.L1Norm,
                "l1": ir.NormKind.L1Norm,
                "l2_norm": ir.NormKind.L2Norm,
                "l2": ir.NormKind.L2Norm,
            }
            if kind_str not in kind_map:
                raise ValueError(f"Unknown tlx.welford_norm kind: {kind_str!r}")
            attr = ir.WelfordNormAttr()
            attr.kind = kind_map[kind_str]
            attr.eps = float(_unwrap_const(kwargs.get("eps", 1e-5)))
            attr.axis = int(_unwrap_const(kwargs.get("axis", -1)))
            attr.num_groups = int(_unwrap_const(kwargs.get("num_groups", 1)))
            attr.stage = int(_unwrap_const(kwargs.get("stage", 0)))
            attr.has_weight = bool(_unwrap_const(kwargs.get("has_weight", False)))
            attr.has_bias = bool(_unwrap_const(kwargs.get("has_bias", False)))
            self.builder.emit_welford_norm([v.vid for v in ssa_ops], attr, self.kernel.hints)
            return None

        if name == "pool":
            if len(args) < 6:
                raise ValueError("tlx.pool requires (x_ptr, out_ptr, N, C, spatial_in..., spatial_out...)")
            ssa_ops = [self._materialize_ssa(v) for v in args]
            for pidx in ssa_ops[1].ptr_params:
                self.kernel.set_param_writable(pidx, True)
            ndim = int(_unwrap_const(kwargs.get("ndim", 2)))
            def _norm_vec(v: Any, default_val: int) -> list[int]:
                u = _unwrap_const(v)
                if u is None:
                    return [default_val] * ndim
                if isinstance(u, int):
                    return [int(u)] * ndim
                return [int(_unwrap_const(e)) for e in u]
            kind_str = str(_unwrap_const(kwargs.get("kind", "max"))).strip().lower()
            attr = ir.PoolAttr()
            attr.kind = ir.PoolKind.Max if kind_str == "max" else ir.PoolKind.Avg
            attr.ndim = ndim
            attr.kernel_size = _norm_vec(kwargs.get("kernel_size", 2), 2)
            attr.stride = _norm_vec(kwargs.get("stride", attr.kernel_size), 1)
            attr.padding = _norm_vec(kwargs.get("padding", 0), 0)
            attr.dilation = _norm_vec(kwargs.get("dilation", 1), 1)
            attr.count_include_pad = bool(_unwrap_const(kwargs.get("count_include_pad", True)))
            attr.ceil_mode = bool(_unwrap_const(kwargs.get("ceil_mode", False)))
            self.builder.emit_pool([v.vid for v in ssa_ops], attr, self.kernel.hints)
            return None

        if name == "row_scan":
            x_ptr = args[0] if len(args) > 0 else kwargs["x_ptr"]
            out_ptr = args[1] if len(args) > 1 else kwargs["out_ptr"]
            num_rows = self._materialize_ssa(args[2] if len(args) > 2 else kwargs["num_rows"])
            row_len = self._materialize_ssa(args[3] if len(args) > 3 else kwargs["row_len"])
            mask_ptr = kwargs.get("mask_ptr")
            if not isinstance(x_ptr, _SSAVal) or not isinstance(out_ptr, _SSAVal):
                raise TypeError("tlx.row_scan expects SSA pointer operands for x_ptr and out_ptr")
            for pidx in out_ptr.ptr_params:
                self.kernel.set_param_writable(pidx, True)
            mask_vid = ir.INVALID_ID
            if mask_ptr is not None:
                if not isinstance(mask_ptr, _SSAVal):
                    raise TypeError("tlx.row_scan mask_ptr must be an SSA pointer")
                mask_vid = mask_ptr.vid
            op = str(_unwrap_const(kwargs.get("op", "add"))).lower()
            attr = ir.ScanAttr()
            attr.kind = ir.ScanKind.Cumsum if op in ("add", "sum", "cumsum") else ir.ScanKind.Cumprod
            attr.axis = -1
            attr.exclusive = bool(_unwrap_const(kwargs.get("exclusive", False)))
            attr.reverse = bool(_unwrap_const(kwargs.get("reverse", False)))
            attr.has_mask = mask_vid != ir.INVALID_ID
            self.builder.emit_row_scan(x_ptr.vid, out_ptr.vid, num_rows.vid, row_len.vid, attr, mask_vid, self.kernel.hints)
            return None

        if name == "implicit_gemm_conv":
            if len(args) < 6:
                raise ValueError("tlx.implicit_gemm_conv requires (x_ptr, w_ptr, out_ptr, [bias_ptr,] dims...)")
            ssa_ops = [self._materialize_ssa(v) for v in args]
            for pidx in ssa_ops[2].ptr_params:
                self.kernel.set_param_writable(pidx, True)
            ndim = int(_unwrap_const(kwargs.get("ndim", 2)))
            def _conv_vec(v: Any, default_val: int) -> list[int]:
                u = _unwrap_const(v)
                if u is None:
                    return [default_val] * ndim
                if isinstance(u, int):
                    return [int(u)] * ndim
                return [int(_unwrap_const(e)) for e in u]
            attr = ir.ConvAttr()
            attr.ndim = ndim
            attr.kernel_size = _conv_vec(kwargs.get("kernel_size", 3), 3)
            attr.stride = _conv_vec(kwargs.get("stride", 1), 1)
            attr.padding = _conv_vec(kwargs.get("padding", 0), 0)
            attr.dilation = _conv_vec(kwargs.get("dilation", 1), 1)
            attr.output_padding = _conv_vec(kwargs.get("output_padding", 0), 0)
            attr.groups = int(_unwrap_const(kwargs.get("groups", 1)))
            attr.transposed = bool(_unwrap_const(kwargs.get("transposed", False)))
            attr.has_bias = bool(_unwrap_const(kwargs.get("has_bias", False)))
            attr.is_depthwise = bool(_unwrap_const(kwargs.get("is_depthwise", False)))
            attr.is_pointwise = bool(_unwrap_const(kwargs.get("is_pointwise", False)))
            attr.layout = _parse_layout_hint(str(_unwrap_const(kwargs.get("layout", "nchw"))))
            self.builder.emit_implicit_gemm_conv([v.vid for v in ssa_ops], attr, self.kernel.hints)
            return None

        if name == "fused_loss":
            kind_str = str(_unwrap_const(args[0] if len(args) > 0 else kwargs["kind"])).strip().lower()
            raw_ops = list(args[1:]) if len(args) > 1 else list(kwargs.get("operands", ()))
            if len(raw_ops) < 4:
                raise ValueError("tlx.fused_loss requires at least 4 operands after kind")
            ssa_ops = [self._materialize_ssa(v) for v in raw_ops]
            kind_map = {
                "mse": ir.LossKind.MSE,
                "cross_entropy": ir.LossKind.CrossEntropy,
                "ce": ir.LossKind.CrossEntropy,
                "huber": ir.LossKind.Huber,
                "smooth_l1": ir.LossKind.Huber,
                "kl_div": ir.LossKind.KLDiv,
                "kldiv": ir.LossKind.KLDiv,
                "triplet_margin": ir.LossKind.TripletMargin,
                "triplet": ir.LossKind.TripletMargin,
                "hinge": ir.LossKind.Hinge,
            }
            if kind_str not in kind_map:
                raise ValueError(f"Unknown tlx.fused_loss kind: {kind_str!r}")
            stage = int(_unwrap_const(kwargs.get("stage", 1)))
            out_idx = 1 if stage == 2 else (3 if kind_map[kind_str] == ir.LossKind.TripletMargin else 2)
            for pidx in ssa_ops[out_idx].ptr_params:
                self.kernel.set_param_writable(pidx, True)
            attr = ir.LossAttr()
            attr.kind = kind_map[kind_str]
            attr.margin = float(_unwrap_const(kwargs.get("margin", 1.0)))
            attr.beta = float(_unwrap_const(kwargs.get("beta", 1.0)))
            attr.eps = float(_unwrap_const(kwargs.get("eps", 1e-6)))
            attr.stage = stage
            self.builder.emit_fused_loss([v.vid for v in ssa_ops], attr, self.kernel.hints)
            return None

        if name == "flash_attention_tile":
            if len(args) < 8:
                raise ValueError("tlx.flash_attention_tile requires (q_ptr, k_ptr, v_ptr, out_ptr, BH, L_q, L_kv, D)")
            ssa_ops = [self._materialize_ssa(v) for v in args[:8]]
            for pidx in ssa_ops[3].ptr_params:
                self.kernel.set_param_writable(pidx, True)
            attr = ir.FlashAttnAttr()
            attr.mode = ir.FlashAttnMode.Fwd
            attr.scale = float(_unwrap_const(kwargs.get("scale", 1.0)))
            attr.causal = bool(_unwrap_const(kwargs.get("causal", False)))
            attr.causal_offset = int(_unwrap_const(kwargs.get("causal_offset", 0)))
            attr.window_size = int(_unwrap_const(kwargs.get("window_size", -1)))
            attr.softcap = float(_unwrap_const(kwargs.get("softcap", 0.0)))
            attr.kv_group_size = int(_unwrap_const(kwargs.get("kv_group_size", 1)))
            lse_ptr = kwargs.get("lse_ptr")
            if lse_ptr is not None:
                lse_ssa = self._materialize_ssa(lse_ptr)
                for pidx in lse_ssa.ptr_params:
                    self.kernel.set_param_writable(pidx, True)
                attr.return_lse = True
                ssa_ops.append(lse_ssa)
            alibi_ptr = kwargs.get("alibi_ptr")
            if alibi_ptr is not None:
                attr.has_alibi = True
                ssa_ops.append(self._materialize_ssa(alibi_ptr))
            if attr.kv_group_size > 1:
                num_q_heads = kwargs.get("num_q_heads")
                if num_q_heads is None:
                    raise ValueError("tlx.flash_attention_tile requires num_q_heads when kv_group_size > 1")
                ssa_ops.append(self._materialize_ssa(num_q_heads))
            self.builder.emit_flash_attention([v.vid for v in ssa_ops], attr, self.kernel.hints)
            return None

        if name == "flash_attention_bwd_dkdv":
            if len(args) < 12:
                raise ValueError(
                    "tlx.flash_attention_bwd_dkdv requires (Q, K, V, dO, L, Delta, dK, dV, BH, Lq, Lkv, D)"
                )
            ssa_ops = [self._materialize_ssa(v) for v in args[:12]]
            for w_idx in (6, 7):
                for pidx in ssa_ops[w_idx].ptr_params:
                    self.kernel.set_param_writable(pidx, True)
            attr = ir.FlashAttnAttr()
            attr.mode = ir.FlashAttnMode.BwdDKDV
            attr.scale = float(_unwrap_const(kwargs.get("scale", 1.0)))
            attr.causal = bool(_unwrap_const(kwargs.get("causal", False)))
            attr.causal_offset = int(_unwrap_const(kwargs.get("causal_offset", 0)))
            attr.window_size = int(_unwrap_const(kwargs.get("window_size", -1)))
            attr.softcap = float(_unwrap_const(kwargs.get("softcap", 0.0)))
            self.builder.emit_flash_attention([v.vid for v in ssa_ops], attr, self.kernel.hints)
            return None

        if name == "flash_attention_bwd_dq":
            if len(args) < 11:
                raise ValueError(
                    "tlx.flash_attention_bwd_dq requires (Q, K, V, dO, L, Delta, dQ, BH, Lq, Lkv, D)"
                )
            ssa_ops = [self._materialize_ssa(v) for v in args[:11]]
            for pidx in ssa_ops[6].ptr_params:
                self.kernel.set_param_writable(pidx, True)
            attr = ir.FlashAttnAttr()
            attr.mode = ir.FlashAttnMode.BwdDQ
            attr.scale = float(_unwrap_const(kwargs.get("scale", 1.0)))
            attr.causal = bool(_unwrap_const(kwargs.get("causal", False)))
            attr.causal_offset = int(_unwrap_const(kwargs.get("causal_offset", 0)))
            attr.window_size = int(_unwrap_const(kwargs.get("window_size", -1)))
            attr.softcap = float(_unwrap_const(kwargs.get("softcap", 0.0)))
            self.builder.emit_flash_attention([v.vid for v in ssa_ops], attr, self.kernel.hints)
            return None

        if name == "paged_attention_decode":
            if len(args) < 12:
                raise ValueError(
                    "tlx.paged_attention_decode requires (Q, K_cache, V_cache, BlockTables, ContextLens, PartialO, PartialLSE, B, H_q, H_kv, MaxBlocks, D)"
                )
            ssa_ops = [self._materialize_ssa(v) for v in args[:12]]
            for w_idx in (5, 6):
                for pidx in ssa_ops[w_idx].ptr_params:
                    self.kernel.set_param_writable(pidx, True)
            attr = ir.FlashAttnAttr()
            attr.mode = ir.FlashAttnMode.PagedDecode
            attr.scale = float(_unwrap_const(kwargs.get("scale", 1.0)))
            attr.softcap = float(_unwrap_const(kwargs.get("softcap", 0.0)))
            self.builder.emit_flash_attention([v.vid for v in ssa_ops], attr, self.kernel.hints)
            return None

        if name == "grouped_gemm":
            if len(args) < 7:
                raise ValueError("tlx.grouped_gemm requires (A, W, Offsets, Out, E, N, K)")
            ssa_ops = [self._materialize_ssa(v) for v in args[:7]]
            for pidx in ssa_ops[3].ptr_params:
                self.kernel.set_param_writable(pidx, True)
            attr = ir.GroupedGemmAttr()
            attr.bm = int(_unwrap_const(kwargs.get("bm", 64)))
            attr.bn = int(_unwrap_const(kwargs.get("bn", 64)))
            attr.bk = int(_unwrap_const(kwargs.get("bk", 16)))
            attr.num_simdgroups = int(_unwrap_const(kwargs.get("num_simdgroups", 4)))
            attr.transpose_b = bool(_unwrap_const(kwargs.get("transpose_b", True)))
            attr.persistent = bool(_unwrap_const(kwargs.get("persistent", True)))
            attr.max_tgs_per_core = int(_unwrap_const(kwargs.get("max_tgs_per_core", 2)))
            self.builder.emit_grouped_gemm([v.vid for v in ssa_ops], attr, self.kernel.hints)
            return None

        if name == "quant_dot":
            if len(args) < 8:
                raise ValueError("tlx.quant_dot requires (A, QWeight, Scales, Zeros, Out, M, N, K)")
            ssa_ops = [self._materialize_ssa(v) for v in args[:8]]
            for pidx in ssa_ops[4].ptr_params:
                self.kernel.set_param_writable(pidx, True)
            attr = ir.QuantDotAttr()
            attr.bits = int(_unwrap_const(kwargs.get("bits", 4)))
            attr.group_size = int(_unwrap_const(kwargs.get("group_size", 128)))
            attr.bm = int(_unwrap_const(kwargs.get("bm", 64)))
            attr.bn = int(_unwrap_const(kwargs.get("bn", 64)))
            attr.bk = int(_unwrap_const(kwargs.get("bk", 32)))
            attr.num_simdgroups = int(_unwrap_const(kwargs.get("num_simdgroups", 4)))
            attr.symmetric = bool(_unwrap_const(kwargs.get("symmetric", False)))
            self.builder.emit_quant_dot([v.vid for v in ssa_ops], attr, self.kernel.hints)
            return None

        if name == "simd_scan":
            x = self._materialize_ssa(args[0] if args else kwargs["x"])
            op = str(kwargs.get("op", "add")).lower()
            axis = int(kwargs.get("axis", -1))
            exclusive = bool(kwargs.get("exclusive", False))
            reverse = bool(kwargs.get("reverse", False))
            mask = kwargs.get("mask")
            mask_vid = ir.INVALID_ID
            if mask is not None:
                mask_ssa = self._ensure_bool(mask)
                x_shape = self.kernel.value_type(x.vid).shape()
                if x_shape:
                    mask_ssa = self._broadcast_to_shape(mask_ssa, x_shape)
                mask_vid = mask_ssa.vid
            attr = ir.ScanAttr()
            attr.kind = ir.ScanKind.Cumsum if op in ("add", "sum", "cumsum") else ir.ScanKind.Cumprod
            attr.axis = axis
            attr.exclusive = exclusive
            attr.reverse = reverse
            attr.has_mask = mask_vid != ir.INVALID_ID
            return _SSAVal(self.builder.emit_scan(x.vid, attr, mask_vid))

        raise NotImplementedError(f"tlx.{name} is scheduled for a later milestone pass and not yet lowered in M1")


def lower_function_to_ir(
    fn: Any,
    param_specs: Sequence[ParamSpec],
    *,
    func_def: ast.FunctionDef | None = None,
    kernel_name: str | None = None,
    hints: Sequence[Any] = (),
) -> _C.ir.Module:
    """Lower a Python `@jit` function with concrete `ParamSpec`s into a C++ `ir.Module`."""
    if func_def is None:
        func_def, _ = parse_kernel_ast(fn)
    name = kernel_name or fn.__name__
    lowerer = _ASTIRLowerer(fn, func_def, param_specs, name, initial_hints=hints)
    return lowerer.lower()
