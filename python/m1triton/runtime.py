"""Runtime bridge to the native Objective-C++ Metal backend."""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Sequence

from m1triton import _C


def device_info():
    return _C.device_info()


def target_spec():
    return _C.TargetSpec.apple_m1_max()


def get_dispatch_count() -> int:
    return int(_C.get_dispatch_count())


def reset_dispatch_count() -> None:
    _C.reset_dispatch_count()


def get_last_kernel_name() -> str:
    return str(_C.get_last_kernel_name())


def synchronize() -> None:
    _C.synchronize()


def compile_msl_smoke(source: str, fn_name: str) -> bool:
    return bool(_C.compile_msl_smoke(source, fn_name))


def compile_kernel(msl_source: str, fn_name: str, *, cache_key: str | None = None, fast_math: bool = False) -> int:
    if cache_key is None:
        digest = hashlib.sha256(f"{fn_name}\n{int(fast_math)}\n{msl_source}".encode("utf-8")).hexdigest()
        cache_key = f"{fn_name}_{digest[:24]}"
    return int(_C.compile_kernel(cache_key, msl_source, fn_name, fast_math))


def make_buffer_arg(tensor_or_array: Any, index: int, *, writable: bool = False) -> _C.BufferArg:
    arg = _C.BufferArg()
    arg.index = index
    arg.writable = writable

    if hasattr(tensor_or_array, "untyped_storage") and hasattr(tensor_or_array, "device"):
        # PyTorch tensor: MPS tensors expose id<MTLBuffer> via untyped_storage().data_ptr()
        # and byte offset via storage_offset() * element_size().
        is_mps = tensor_or_array.device.type == "mps"
        arg.is_metal_buffer = is_mps
        if is_mps:
            arg.handle = int(tensor_or_array.untyped_storage().data_ptr())
            arg.byte_offset = int(tensor_or_array.storage_offset()) * int(tensor_or_array.element_size())
            arg.nbytes = int(tensor_or_array.numel()) * int(tensor_or_array.element_size())
        else:
            if not tensor_or_array.is_contiguous():
                raise ValueError("Non-contiguous CPU tensor views must be made contiguous before Metal dispatch")
            arg.handle = int(tensor_or_array.data_ptr())
            arg.byte_offset = 0
            arg.nbytes = int(tensor_or_array.numel()) * int(tensor_or_array.element_size())
        return arg

    if hasattr(tensor_or_array, "__array_interface__"):
        arg.is_metal_buffer = False
        arg.handle = int(tensor_or_array.__array_interface__["data"][0])
        arg.byte_offset = 0
        arg.nbytes = int(tensor_or_array.nbytes)
        return arg

    raise TypeError(f"Unsupported buffer argument type: {type(tensor_or_array)}")


def make_scalar_arg(value: int | float | bool, index: int, dtype: str = "i32") -> _C.ScalarArg:
    arg = _C.ScalarArg()
    arg.index = index
    if dtype == "i32":
        raw = struct.pack("<i", int(value))
    elif dtype == "u32":
        raw = struct.pack("<I", int(value))
    elif dtype == "i64":
        raw = struct.pack("<q", int(value))
    elif dtype == "u64":
        raw = struct.pack("<Q", int(value))
    elif dtype == "f32":
        raw = struct.pack("<f", float(value))
    else:
        raise ValueError(f"Unsupported scalar dtype: {dtype}")
    arg.bytes = list(raw)
    return arg


def launch_kernel(
    kernel_id: int,
    buffers: Sequence[_C.BufferArg],
    scalars: Sequence[_C.ScalarArg],
    *,
    grid: tuple[int, int, int],
    threads_per_threadgroup: tuple[int, int, int],
    threadgroup_mem_bytes: int = 0,
    wait: bool = False,
) -> float:
    cfg = _C.LaunchConfig()
    cfg.grid_x, cfg.grid_y, cfg.grid_z = int(grid[0]), int(grid[1]), int(grid[2])
    cfg.tg_x, cfg.tg_y, cfg.tg_z = (
        int(threads_per_threadgroup[0]),
        int(threads_per_threadgroup[1]),
        int(threads_per_threadgroup[2]),
    )
    cfg.threadgroup_mem_bytes = int(threadgroup_mem_bytes)
    cfg.wait = bool(wait)
    return float(_C.launch_kernel(int(kernel_id), list(buffers), list(scalars), cfg))
