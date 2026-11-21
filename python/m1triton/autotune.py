"""Roofline-guided `@m1triton.autotune` and `Config` for Apple M1 Max."""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Sequence

from m1triton import runtime as rt
from m1triton.jit import CompiledKernel, JITFunction, _is_buffer_arg


class Config:
    """Autotuning candidate schedule configuration compatible with `triton.Config`."""

    def __init__(
        self,
        kwargs: dict[str, Any],
        *,
        num_warps: int = 4,
        num_stages: int = 1,
        pre_hook: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.kwargs = dict(kwargs)
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)
        self.pre_hook = pre_hook

    def all_kwargs(self) -> dict[str, Any]:
        merged = dict(self.kwargs)
        merged["num_warps"] = self.num_warps
        merged["num_stages"] = self.num_stages
        return merged

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"Config({{{items}}}, num_warps={self.num_warps}, num_stages={self.num_stages})"


def _estimate_config_tg_bytes(cfg: Config, elem_bytes: int) -> int:
    kw = cfg.kwargs
    if "threadgroup_mem_bytes" in kw:
        return int(kw["threadgroup_mem_bytes"])

    bm = kw.get("BLOCK_M", kw.get("BM"))
    bn = kw.get("BLOCK_N", kw.get("BN"))
    bk = kw.get("BLOCK_K", kw.get("BK"))
    if bm is not None and bn is not None and bk is not None:
        stages = max(1, int(cfg.num_stages))
        if kw.get("double_buffer"):
            stages = max(stages, 2)
        return (int(bm) * int(bk) + int(bk) * int(bn)) * elem_bytes * stages

    return 0


def roofline_filter_configs(
    configs: Sequence[Config],
    *,
    elem_bytes: int = 4,
    target: Any = None,
) -> list[Config]:
    """Prune schedules violating M1 Max threadgroup size or 32 KB threadgroup SRAM limits."""
    if not configs:
        return []
    spec = target if target is not None else rt.target_spec()
    simd_w = int(spec.simd_width)
    max_threads = int(spec.max_threads_per_threadgroup)

    feasible: list[Config] = []
    for cfg in configs:
        threads = max(1, cfg.num_warps) * simd_w
        if threads > max_threads:
            continue
        est_bytes = _estimate_config_tg_bytes(cfg, elem_bytes)
        if not spec.fits_threadgroup_memory(est_bytes):
            continue
        feasible.append(cfg)

    if feasible:
        return feasible
    return [min(configs, key=lambda c: _estimate_config_tg_bytes(c, elem_bytes))]


class _AutotuneGridLauncher:
    def __init__(self, tuner: Autotuner, grid: Any) -> None:
        self._tuner = tuner
        self._grid = grid

    def __call__(self, *args: Any, **kwargs: Any) -> CompiledKernel:
        return self._tuner._run(self._grid, args, kwargs)


class Autotuner:
    """Autotuning wrapper around a `JITFunction` with M1 Max roofline pre-filtering and timing cache."""

    def __init__(
        self,
        fn: JITFunction,
        configs: Sequence[Config],
        key: Sequence[str],
        *,
        warmup: int = 3,
        rep: int = 10,
        prune_configs_by: dict[str, Any] | None = None,
        reset_to_zero: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(fn, JITFunction):
            raise TypeError("@autotune must decorate an @m1triton.jit function")
        if not configs:
            raise ValueError("@autotune requires at least one Config")
        self.fn = fn
        self.configs = list(configs)
        self.key = tuple(key)
        self.warmup = max(1, int(warmup))
        self.rep = max(1, int(rep))
        self.prune_configs_by = prune_configs_by or {}
        self.reset_to_zero = tuple(reset_to_zero or ())
        self.cache: dict[tuple[Any, ...], Config] = {}
        self.best_config: Config | None = None
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def clear_cache(self) -> None:
        self.cache.clear()
        self.best_config = None

    def __getitem__(self, grid: Any) -> _AutotuneGridLauncher:
        return _AutotuneGridLauncher(self, grid)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"Autotuned kernel '{self.__name__}' must be launched with grid syntax: `{self.__name__}[grid](*args)`."
        )

    def _named_args(self, args: Sequence[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        named: dict[str, Any] = {}
        for name, val in zip(self.fn._param_names, args):
            named[name] = val
        named.update(kwargs)
        return named

    def _infer_elem_bytes(self, args: Sequence[Any], kwargs: dict[str, Any]) -> int:
        for val in list(args) + list(kwargs.values()):
            if _is_buffer_arg(val):
                if hasattr(val, "element_size"):
                    return int(val.element_size())
                if hasattr(val, "itemsize"):
                    return int(val.itemsize)
        return 4

    def _build_cache_key(self, named_args: dict[str, Any]) -> tuple[Any, ...]:
        key_vals = tuple(named_args.get(k) for k in self.key)
        dtype_tags = tuple(
            (k, str(v.dtype))
            for k, v in sorted(named_args.items())
            if _is_buffer_arg(v)
        )
        return (key_vals, dtype_tags)

    def _bench_config(
        self,
        cfg: Config,
        grid: Any,
        args: Sequence[Any],
        base_kwargs: dict[str, Any],
        named_args: dict[str, Any],
    ) -> float:
        merged_kwargs = dict(base_kwargs)
        merged_kwargs.update(cfg.all_kwargs())
        merged_kwargs["wait"] = True

        def _reset_buffers() -> None:
            if cfg.pre_hook is not None:
                cfg.pre_hook(named_args)
            for name in self.reset_to_zero:
                buf = named_args.get(name)
                if buf is not None and hasattr(buf, "zero_"):
                    buf.zero_()

        for _ in range(self.warmup):
            _reset_buffers()
            self.fn[grid](*args, **merged_kwargs)

        times_ms: list[float] = []
        for _ in range(self.rep):
            _reset_buffers()
            t0 = time.perf_counter()
            compiled = self.fn[grid](*args, **merged_kwargs)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            gpu_ms = compiled.last_gpu_ms if compiled.last_gpu_ms > 0.0 else wall_ms
            times_ms.append(gpu_ms)
        return float(statistics.median(times_ms))

    def _select_config(
        self,
        grid: Any,
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> Config:
        named_args = self._named_args(args, kwargs)
        ckey = self._build_cache_key(named_args)
        cached = self.cache.get(ckey)
        if cached is not None:
            self.best_config = cached
            return cached

        elem_bytes = self._infer_elem_bytes(args, kwargs)
        candidates = roofline_filter_configs(self.configs, elem_bytes=elem_bytes)

        early_prune = self.prune_configs_by.get("early_config_prune")
        if callable(early_prune):
            pruned = early_prune(candidates, named_args)
            if pruned:
                candidates = list(pruned)

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            best_cfg: Config | None = None
            best_ms = float("inf")
            last_err: Exception | None = None
            for cfg in candidates:
                try:
                    med_ms = self._bench_config(cfg, grid, args, kwargs, named_args)
                except Exception as exc:
                    last_err = exc
                    continue
                if med_ms < best_ms:
                    best_ms = med_ms
                    best_cfg = cfg
            if best_cfg is None:
                raise RuntimeError(f"All autotune configs failed for '{self.__name__}'") from last_err
            chosen = best_cfg

        self.cache[ckey] = chosen
        self.best_config = chosen
        return chosen

    def _run(self, grid: Any, args: Sequence[Any], kwargs: dict[str, Any]) -> CompiledKernel:
        named_args = self._named_args(args, kwargs)
        ckey = self._build_cache_key(named_args)
        was_cached = ckey in self.cache
        cfg = self._select_config(grid, args, kwargs)
        if not was_cached:
            if cfg.pre_hook is not None:
                cfg.pre_hook(named_args)
            for name in self.reset_to_zero:
                buf = named_args.get(name)
                if buf is not None and hasattr(buf, "zero_"):
                    buf.zero_()
        merged = dict(kwargs)
        merged.update(cfg.all_kwargs())
        return self.fn[grid](*args, **merged)

    def inspect_ir(self, *args: Any, **kwargs: Any) -> tuple[str, str]:
        cfg = self.best_config or roofline_filter_configs(self.configs)[0]
        merged = dict(kwargs)
        merged.update(cfg.all_kwargs())
        return self.fn.inspect_ir(*args, **merged)

    def inspect_msl(self, *args: Any, **kwargs: Any) -> str:
        cfg = self.best_config or roofline_filter_configs(self.configs)[0]
        merged = dict(kwargs)
        merged.update(cfg.all_kwargs())
        return self.fn.inspect_msl(*args, **merged)


def autotune(
    configs: Sequence[Config],
    key: Sequence[str],
    *,
    warmup: int = 3,
    rep: int = 10,
    prune_configs_by: dict[str, Any] | None = None,
    reset_to_zero: Sequence[str] | None = None,
) -> Callable[[JITFunction], Autotuner]:
    """Decorator attaching roofline-pruned GPU autotuning over an `@m1triton.jit` kernel."""

    def decorator(fn: JITFunction) -> Autotuner:
        return Autotuner(
            fn,
            configs=configs,
            key=key,
            warmup=warmup,
            rep=rep,
            prune_configs_by=prune_configs_by,
            reset_to_zero=reset_to_zero,
        )

    return decorator


__all__ = ["Autotuner", "Config", "autotune", "roofline_filter_configs"]
