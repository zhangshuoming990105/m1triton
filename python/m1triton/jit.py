"""`@m1triton.jit` decorator, specialization cache, IR/MSL inspection, and UMA Metal dispatch."""

from __future__ import annotations

import hashlib
import inspect
import os
import struct
import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from m1triton import _C
from m1triton import language as tl
from m1triton import runtime as rt
from m1triton.ast_lowering import ParamSpec, lower_function_to_ir, parse_kernel_ast

_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1

_LAUNCH_ONLY_KWARGS = frozenset(
    {
        "num_warps",
        "num_stages",
        "num_ctas",
        "fast_math",
        "dump_ir",
        "dump_msl",
        "verify_ir",
        "wait",
        "sync_mps",
    }
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _is_buffer_arg(val: Any) -> bool:
    if hasattr(val, "untyped_storage") and hasattr(val, "device") and hasattr(val, "dtype"):
        return True
    if hasattr(val, "__array_interface__") and hasattr(val, "dtype") and getattr(val, "ndim", 0) > 0:
        return True
    return False


def _is_mps_tensor(val: Any) -> bool:
    return hasattr(val, "device") and getattr(val.device, "type", None) == "mps"


def _is_ptr_16b_aligned(val: Any) -> bool:
    if hasattr(val, "untyped_storage") and hasattr(val, "device"):
        if _is_mps_tensor(val):
            byte_offset = int(val.storage_offset()) * int(val.element_size())
            return (byte_offset % 16) == 0
        return (int(val.data_ptr()) % 16) == 0
    if hasattr(val, "__array_interface__"):
        return (int(val.__array_interface__["data"][0]) % 16) == 0
    return False


def _dtype_from_tensor_or_scalar_dtype(dtype_obj: Any) -> tl.dtype:
    name = str(dtype_obj)
    if "." in name:
        name = name.split(".")[-1]
    mapping = {
        "float32": tl.float32,
        "float": tl.float32,
        "float16": tl.float16,
        "half": tl.float16,
        "int32": tl.int32,
        "uint32": tl.uint32,
        "int64": tl.int64,
        "bool": tl.bool_,
        "bool_": tl.bool_,
    }
    if name not in mapping:
        raise TypeError(f"Unsupported tensor/scalar dtype for M1Triton kernel: {dtype_obj!r}")
    return mapping[name]


def _infer_scalar_dtype(val: Any, annotation: Any = None) -> tl.dtype:
    if isinstance(annotation, tl.dtype):
        return annotation
    if isinstance(val, bool):
        return tl.bool_
    if isinstance(val, int):
        return tl.int32 if _INT32_MIN <= val <= _INT32_MAX else tl.int64
    if isinstance(val, float):
        return tl.float32
    if hasattr(val, "dtype"):
        return _dtype_from_tensor_or_scalar_dtype(val.dtype)
    raise TypeError(f"Unsupported runtime scalar argument type: {type(val)} ({val!r})")


def _freeze_constexpr(val: Any) -> Any:
    if isinstance(val, tl.constexpr):
        val = val.value
    if isinstance(val, (bool, int, float, str, bytes, type(None), tl.dtype)):
        return (type(val).__name__, val)
    if isinstance(val, (tuple, list)):
        return tuple(_freeze_constexpr(x) for x in val)
    return repr(val)


def _is_constexpr_param(param: inspect.Parameter, value: Any) -> bool:
    if isinstance(value, tl.constexpr):
        return True
    ann = param.annotation
    if ann is tl.constexpr:
        return True
    if isinstance(ann, str) and ann.strip().split(".")[-1] == "constexpr":
        return True
    return False


def _pack_scalar_arg(value: Any, index: int, ir_dtype: _C.ir.ScalarDType) -> _C.ScalarArg:
    ir = _C.ir
    if ir_dtype == ir.ScalarDType.I32:
        return rt.make_scalar_arg(int(value), index, "i32")
    if ir_dtype == ir.ScalarDType.U32:
        return rt.make_scalar_arg(int(value), index, "u32")
    if ir_dtype == ir.ScalarDType.I64:
        return rt.make_scalar_arg(int(value), index, "i64")
    if ir_dtype == ir.ScalarDType.F32:
        return rt.make_scalar_arg(float(value), index, "f32")
    if ir_dtype == ir.ScalarDType.F16:
        arg = _C.ScalarArg()
        arg.index = index
        arg.bytes = list(struct.pack("<e", float(value)))
        return arg
    if ir_dtype == ir.ScalarDType.Bool:
        arg = _C.ScalarArg()
        arg.index = index
        arg.bytes = list(struct.pack("<?", bool(value)))
        return arg
    raise ValueError(f"Unsupported IR scalar dtype for launch: {ir_dtype!r}")


def _normalize_grid(grid: Any, meta: dict[str, Any]) -> tuple[int, int, int]:
    raw = grid(meta) if callable(grid) else grid
    if isinstance(raw, int):
        return (int(raw), 1, 1)
    dims = tuple(int(d) for d in raw)
    if len(dims) == 1:
        return (dims[0], 1, 1)
    if len(dims) == 2:
        return (dims[0], dims[1], 1)
    if len(dims) == 3:
        return (dims[0], dims[1], dims[2])
    raise ValueError(f"Grid must have 1 to 3 dimensions, got {dims!r}")


@dataclass
class CompiledKernel:
    """Cached compiled M1Triton specialization artifact and Metal pipeline handle."""

    kernel_id: int
    cache_key: tuple[Any, ...]
    pre_ir: str
    post_ir: str
    msl_source: str
    launch_spec: _C.LaunchSpec
    fast_math: bool
    last_gpu_ms: float = 0.0


class _GridLauncher:
    """Callable returned by `kernel[grid]` to specialize and launch on Metal."""

    def __init__(self, jit_fn: JITFunction, grid: Any) -> None:
        self._jit_fn = jit_fn
        self._grid = grid

    def __call__(self, *args: Any, **kwargs: Any) -> CompiledKernel:
        return self._jit_fn._launch(self._grid, args, kwargs)


class JITFunction:
    """Compiled Triton-compatible / Co-designed kernel backed by M1Triton C++ SSA IR and Metal."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        fast_math: bool = False,
        dump_ir: bool = False,
        dump_msl: bool = False,
        verify_ir: bool = True,
        hints: Sequence[Any] = (),
    ) -> None:
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__module__ = fn.__module__
        self.__qualname__ = getattr(fn, "__qualname__", fn.__name__)

        self.fast_math = bool(fast_math)
        self.dump_ir = bool(dump_ir)
        self.dump_msl = bool(dump_msl)
        self.verify_ir = bool(verify_ir)
        self.hints = tuple(hints)

        self._sig = inspect.signature(fn)
        self._param_names = tuple(self._sig.parameters.keys())
        self._param_name_set = frozenset(self._param_names)
        self._func_def, self._source = parse_kernel_ast(fn)
        self._source_hash = hashlib.sha256(self._source.encode("utf-8")).hexdigest()
        self._fn_id = f"{self.__module__}.{self.__qualname__}@{id(fn)}"
        self._cache: dict[tuple[Any, ...], CompiledKernel] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    def __getitem__(self, grid: Any) -> _GridLauncher:
        return _GridLauncher(self, grid)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"M1Triton kernel '{self.__name__}' must be launched with grid syntax: `{self.__name__}[grid](*args)`."
        )

    def _split_launch_kwargs(self, kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        kernel_kwargs: dict[str, Any] = {}
        launch_opts: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in _LAUNCH_ONLY_KWARGS and k not in self._param_name_set:
                launch_opts[k] = v
            else:
                kernel_kwargs[k] = v
        return kernel_kwargs, launch_opts

    def _prepare_specialization(
        self,
        args: Sequence[Any],
        kernel_kwargs: dict[str, Any],
        launch_opts: dict[str, Any],
    ) -> tuple[
        tuple[Any, ...],
        list[ParamSpec],
        dict[str, Any],
        list[Any],
        dict[str, Any],
        bool,
    ]:
        bound = self._sig.bind(*args, **kernel_kwargs)
        bound.apply_defaults()

        param_specs: list[ParamSpec] = []
        constexpr_map: dict[str, Any] = {}
        runtime_by_name: dict[str, Any] = {}
        runtime_ordered: list[Any] = []
        arg_dtypes: list[tuple[str, str]] = []
        ptr_aligned_flags: list[bool] = []

        for name, param in self._sig.parameters.items():
            val = bound.arguments[name]
            if _is_constexpr_param(param, val):
                cval = val.value if isinstance(val, tl.constexpr) else val
                constexpr_map[name] = cval
                param_specs.append(ParamSpec(name=name, is_constexpr=True, constexpr_val=cval))
            elif _is_buffer_arg(val):
                dt = _dtype_from_tensor_or_scalar_dtype(val.dtype)
                aligned = _is_ptr_16b_aligned(val)
                ptr_aligned_flags.append(aligned)
                arg_dtypes.append((name, f"ptr:{dt.name}"))
                runtime_by_name[name] = val
                runtime_ordered.append(val)
                param_specs.append(
                    ParamSpec(
                        name=name,
                        is_constexpr=False,
                        is_pointer=True,
                        dtype=dt,
                        divisibility=16 if aligned else 1,
                    )
                )
            else:
                dt = _infer_scalar_dtype(val, param.annotation)
                arg_dtypes.append((name, f"scalar:{dt.name}"))
                runtime_by_name[name] = val
                runtime_ordered.append(val)
                param_specs.append(
                    ParamSpec(
                        name=name,
                        is_constexpr=False,
                        is_pointer=False,
                        dtype=dt,
                        divisibility=1,
                    )
                )

        effective_fast_math = bool(launch_opts.get("fast_math", self.fast_math))
        sorted_constexprs = tuple(sorted((k, _freeze_constexpr(v)) for k, v in constexpr_map.items()))
        cache_key = (
            self._fn_id,
            self._source_hash,
            sorted_constexprs,
            tuple(arg_dtypes),
            tuple(ptr_aligned_flags),
            effective_fast_math,
        )
        return (
            cache_key,
            param_specs,
            runtime_by_name,
            runtime_ordered,
            constexpr_map,
            effective_fast_math,
        )

    def _compile_specialized(
        self,
        cache_key: tuple[Any, ...],
        param_specs: Sequence[ParamSpec],
        effective_fast_math: bool,
        *,
        dump_ir: bool,
        dump_msl: bool,
        verify_ir: bool,
        compile_metal: bool = True,
    ) -> CompiledKernel:
        cached = self._cache.get(cache_key)
        if cached is not None and (not compile_metal or cached.kernel_id != 0):
            if dump_ir:
                sys.stderr.write(
                    f"=== M1Triton Pre-Pass IR (cached) ===\n{cached.pre_ir}\n"
                    f"=== M1Triton Post-Pass IR (cached) ===\n{cached.post_ir}\n"
                )
            if dump_msl:
                sys.stderr.write(
                    f"=== M1Triton Generated MSL (cached: {cached.launch_spec.entry_name}) ===\n"
                    f"{cached.msl_source}\n"
                )
            return cached

        ir_module = lower_function_to_ir(
            self.fn,
            param_specs,
            func_def=self._func_def,
            kernel_name=self.__name__,
            hints=self.hints,
        )
        target = rt.target_spec()
        compiled_msl, pre_ir, post_ir = _C.compile_kernel_ir(
            ir_module,
            target,
            verify_ir=verify_ir,
            dump_ir=dump_ir,
            dump_msl=dump_msl,
        )
        if not compiled_msl.ok:
            raise RuntimeError(f"Metal codegen failed for kernel '{self.__name__}': {compiled_msl.error}")

        use_fast_math = bool(effective_fast_math or compiled_msl.launch_spec.fast_math)
        kernel_id = 0
        if compile_metal:
            key_digest = hashlib.sha256(
                f"{cache_key!r}\n{int(use_fast_math)}\n{compiled_msl.msl_source}".encode("utf-8")
            ).hexdigest()
            rt_key = f"{compiled_msl.launch_spec.entry_name}_{key_digest[:24]}"
            kernel_id = rt.compile_kernel(
                compiled_msl.msl_source,
                compiled_msl.launch_spec.entry_name,
                cache_key=rt_key,
                fast_math=use_fast_math,
            )

        compiled = CompiledKernel(
            kernel_id=kernel_id,
            cache_key=cache_key,
            pre_ir=pre_ir,
            post_ir=post_ir,
            msl_source=compiled_msl.msl_source,
            launch_spec=compiled_msl.launch_spec,
            fast_math=use_fast_math,
        )
        if compile_metal:
            self._cache[cache_key] = compiled
        return compiled

    def compile(self, *args: Any, **kwargs: Any) -> CompiledKernel:
        """Lower, optimize, codegen, and compile this kernel specialization without launching."""
        kernel_kwargs, launch_opts = self._split_launch_kwargs(kwargs)
        cache_key, param_specs, _, _, _, fast_math = self._prepare_specialization(args, kernel_kwargs, launch_opts)
        dump_ir = bool(launch_opts.get("dump_ir", self.dump_ir or _env_flag("M1TRITON_DUMP_IR")))
        dump_msl = bool(launch_opts.get("dump_msl", self.dump_msl or _env_flag("M1TRITON_DUMP_MSL")))
        verify_ir = bool(launch_opts.get("verify_ir", self.verify_ir))
        return self._compile_specialized(
            cache_key,
            param_specs,
            fast_math,
            dump_ir=dump_ir,
            dump_msl=dump_msl,
            verify_ir=verify_ir,
            compile_metal=True,
        )

    def inspect_ir(self, *args: Any, **kwargs: Any) -> tuple[str, str]:
        """Return `(pre_pass_ir, post_pass_ir)` for the specialization matching `*args, **kwargs`."""
        kernel_kwargs, launch_opts = self._split_launch_kwargs(kwargs)
        cache_key, param_specs, _, _, _, fast_math = self._prepare_specialization(args, kernel_kwargs, launch_opts)
        verify_ir = bool(launch_opts.get("verify_ir", self.verify_ir))
        compiled = self._compile_specialized(
            cache_key,
            param_specs,
            fast_math,
            dump_ir=False,
            dump_msl=False,
            verify_ir=verify_ir,
            compile_metal=False,
        )
        return compiled.pre_ir, compiled.post_ir

    def inspect_msl(self, *args: Any, **kwargs: Any) -> str:
        """Return generated MSL source for the specialization matching `*args, **kwargs`."""
        kernel_kwargs, launch_opts = self._split_launch_kwargs(kwargs)
        cache_key, param_specs, _, _, _, fast_math = self._prepare_specialization(args, kernel_kwargs, launch_opts)
        verify_ir = bool(launch_opts.get("verify_ir", self.verify_ir))
        compiled = self._compile_specialized(
            cache_key,
            param_specs,
            fast_math,
            dump_ir=False,
            dump_msl=False,
            verify_ir=verify_ir,
            compile_metal=False,
        )
        return compiled.msl_source

    def _launch(self, grid: Any, args: Sequence[Any], kwargs: dict[str, Any]) -> CompiledKernel:
        kernel_kwargs, launch_opts = self._split_launch_kwargs(kwargs)
        (
            cache_key,
            param_specs,
            runtime_by_name,
            runtime_ordered,
            constexpr_map,
            fast_math,
        ) = self._prepare_specialization(args, kernel_kwargs, launch_opts)

        dump_ir = bool(launch_opts.get("dump_ir", self.dump_ir or _env_flag("M1TRITON_DUMP_IR")))
        dump_msl = bool(launch_opts.get("dump_msl", self.dump_msl or _env_flag("M1TRITON_DUMP_MSL")))
        verify_ir = bool(launch_opts.get("verify_ir", self.verify_ir))
        wait = bool(launch_opts.get("wait", False))
        sync_mps = bool(launch_opts.get("sync_mps", True))

        compiled = self._compile_specialized(
            cache_key,
            param_specs,
            fast_math,
            dump_ir=dump_ir,
            dump_msl=dump_msl,
            verify_ir=verify_ir,
            compile_metal=True,
        )

        grid_3d = _normalize_grid(grid, constexpr_map)
        if grid_3d[0] <= 0 or grid_3d[1] <= 0 or grid_3d[2] <= 0:
            compiled.last_gpu_ms = 0.0
            return compiled

        spec = compiled.launch_spec
        buffers: list[_C.BufferArg] = []
        scalars: list[_C.ScalarArg] = []
        has_mps_buffer = False

        for slot in spec.param_slots:
            val = runtime_by_name[slot.name]
            if slot.is_pointer:
                if _is_mps_tensor(val):
                    has_mps_buffer = True
                buffers.append(rt.make_buffer_arg(val, int(slot.buffer_index), writable=bool(slot.writable)))
            else:
                scalars.append(_pack_scalar_arg(val, int(slot.buffer_index), slot.dtype))

        if sync_mps and has_mps_buffer:
            torch_mod = sys.modules.get("torch")
            if torch_mod is not None and hasattr(torch_mod, "mps"):
                torch_mod.mps.synchronize()

        profile_enabled = os.environ.get("M1TRITON_PROFILE", "0") == "1"
        gpu_ms = rt.launch_kernel(
            compiled.kernel_id,
            buffers,
            scalars,
            grid=grid_3d,
            threads_per_threadgroup=(int(spec.tg_x), int(spec.tg_y), int(spec.tg_z)),
            threadgroup_mem_bytes=int(spec.threadgroup_mem_bytes),
            wait=bool(wait or profile_enabled),
        )
        if profile_enabled:
            print(
                f"[M1TRITON_PROFILE] kernel={spec.entry_name} grid={grid_3d} "
                f"tg=({spec.tg_x},{spec.tg_y},{spec.tg_z}) vec={spec.vec_width} gpu_ms={gpu_ms:.4f}",
                file=sys.stderr,
            )
        compiled.last_gpu_ms = gpu_ms
        return compiled


def jit(
    fn: Callable[..., Any] | None = None,
    *,
    fast_math: bool = False,
    dump_ir: bool = False,
    dump_msl: bool = False,
    verify_ir: bool = True,
    hints: Sequence[Any] = (),
) -> JITFunction | Callable[[Callable[..., Any]], JITFunction]:
    """Decorator compiling a Python kernel function through M1Triton SSA IR to Metal."""

    def decorator(target_fn: Callable[..., Any]) -> JITFunction:
        return JITFunction(
            target_fn,
            fast_math=fast_math,
            dump_ir=dump_ir,
            dump_msl=dump_msl,
            verify_ir=verify_ir,
            hints=hints,
        )

    if fn is not None:
        return decorator(fn)
    return decorator


__all__ = ["CompiledKernel", "JITFunction", "jit"]
