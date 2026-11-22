"""Roofline-guided `@m1triton.autotune`, `@m1triton.heuristics`, and persistent cache for Apple M1 Max."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import statistics
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from m1triton import language as tl
from m1triton import runtime as rt
from m1triton.jit import CompiledKernel, JITFunction, _is_buffer_arg, _is_mps_tensor

_DISK_CACHE_VERSION = 1


class _PSOInfeasibleError(RuntimeError):
    """Raised internally when a compiled PSO cannot support a config's threadgroup size."""


def _is_disk_cache_disabled() -> bool:
    return os.environ.get("M1TRITON_DISABLE_AUTOTUNE_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def get_autotune_cache_dir() -> Path:
    """Return the persistent autotune cache directory (`$M1TRITON_CACHE_DIR/autotune` or `~/.cache/m1triton/autotune`)."""
    env_dir = os.environ.get("M1TRITON_CACHE_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser() / "autotune"
    return Path.home() / ".cache" / "m1triton" / "autotune"


def _canonicalize_json_value(val: Any) -> Any:
    if isinstance(val, tl.constexpr):
        val = val.value
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, tl.dtype):
        return f"dtype:{val.name}"
    if isinstance(val, (tuple, list)):
        return [_canonicalize_json_value(item) for item in val]
    if isinstance(val, Mapping):
        return {str(k): _canonicalize_json_value(v) for k, v in sorted(val.items(), key=lambda kv: str(kv[0]))}
    return repr(val)


def _unwrap_jit_function(fn: Any) -> JITFunction:
    cur = fn
    visited: set[int] = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        if isinstance(cur, JITFunction):
            return cur
        cur = getattr(cur, "fn", None)
    raise TypeError("Expected an @m1triton.jit function (optionally wrapped by @autotune or @heuristics)")


def _wrap_grid_with_meta(grid: Any, extra_meta: Mapping[str, Any]) -> Any:
    if not callable(grid) or not extra_meta:
        return grid
    snapshot = dict(extra_meta)

    def _wrapped(meta: dict[str, Any]) -> Any:
        merged = dict(snapshot)
        merged.update(meta)
        return grid(merged)

    return _wrapped


def _get_fn_source_hash(jit_fn: JITFunction) -> str:
    try:
        src = textwrap.dedent(inspect.getsource(jit_fn.fn))
        return hashlib.sha256(src.encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        existing = getattr(jit_fn, "_source_hash", "")
        if existing:
            return str(existing)
        return hashlib.sha256(jit_fn.__name__.encode("utf-8")).hexdigest()


def _get_gpu_name() -> str:
    try:
        info = rt.device_info()
        name = getattr(info, "name", "")
        return str(name) if name else "unknown"
    except Exception:
        return "unknown"


def _sync_mps_if_loaded() -> None:
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and hasattr(torch_mod, "mps") and hasattr(torch_mod.mps, "synchronize"):
        torch_mod.mps.synchronize()


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

    def signature(self) -> dict[str, Any]:
        return {
            "kwargs": _canonicalize_json_value(self.kwargs),
            "num_warps": int(self.num_warps),
            "num_stages": int(self.num_stages),
        }

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


class _HeuristicsGridLauncher:
    def __init__(self, heuristics_fn: Heuristics, grid: Any) -> None:
        self._heuristics = heuristics_fn
        self._grid = grid

    def __call__(self, *args: Any, **kwargs: Any) -> CompiledKernel:
        return self._heuristics._run(self._grid, args, kwargs)


class Heuristics:
    """Triton-compatible `@m1triton.heuristics` wrapper deriving constexpr kwargs before launch."""

    def __init__(
        self,
        fn: JITFunction | Autotuner | Heuristics,
        values: Mapping[str, Callable[[dict[str, Any]], Any]],
    ) -> None:
        self._jit_fn = _unwrap_jit_function(fn)
        if not isinstance(values, Mapping) or not values:
            raise ValueError("@heuristics requires a non-empty mapping of parameter names to callables")
        for key, value_fn in values.items():
            if not isinstance(key, str) or not callable(value_fn):
                raise TypeError(f"@heuristics entry for {key!r} must map a str name to a callable")
        self.fn = fn
        self.values: dict[str, Callable[[dict[str, Any]], Any]] = dict(values)
        self._param_names = self._jit_fn._param_names
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    @property
    def best_config(self) -> Config | None:
        return getattr(self.fn, "best_config", None)

    def clear_cache(self) -> None:
        if hasattr(self.fn, "clear_cache"):
            self.fn.clear_cache()

    def __getitem__(self, grid: Any) -> _HeuristicsGridLauncher:
        return _HeuristicsGridLauncher(self, grid)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"Heuristics kernel '{self.__name__}' must be launched with grid syntax: `{self.__name__}[grid](*args)`."
        )

    def _compute_heuristics(
        self,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        named: dict[str, Any] = {}
        for name, param in self._jit_fn._sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                val = param.default
                named[name] = val.value if isinstance(val, tl.constexpr) else val
        for name, val in zip(self._param_names, args):
            named[name] = val.value if isinstance(val, tl.constexpr) else val
        for name, val in kwargs.items():
            named[name] = val.value if isinstance(val, tl.constexpr) else val

        heur_kwargs: dict[str, Any] = {}
        for name, value_fn in self.values.items():
            computed = value_fn(named)
            unwrapped = computed.value if isinstance(computed, tl.constexpr) else computed
            named[name] = unwrapped
            heur_kwargs[name] = computed

        merged = dict(kwargs)
        merged.update(heur_kwargs)
        return merged, heur_kwargs

    def _run(self, grid: Any, args: Sequence[Any], kwargs: dict[str, Any]) -> CompiledKernel:
        merged, heur_kwargs = self._compute_heuristics(args, kwargs)
        wrapped_grid = _wrap_grid_with_meta(grid, heur_kwargs)
        return self.fn[wrapped_grid](*args, **merged)

    def compile(self, *args: Any, **kwargs: Any) -> CompiledKernel:
        merged, _ = self._compute_heuristics(args, kwargs)
        return self.fn.compile(*args, **merged)

    def inspect_ir(self, *args: Any, **kwargs: Any) -> tuple[str, str]:
        merged, _ = self._compute_heuristics(args, kwargs)
        return self.fn.inspect_ir(*args, **merged)

    def inspect_msl(self, *args: Any, **kwargs: Any) -> str:
        merged, _ = self._compute_heuristics(args, kwargs)
        return self.fn.inspect_msl(*args, **merged)


def heuristics(
    values: Mapping[str, Callable[[dict[str, Any]], Any]],
) -> Callable[[JITFunction | Autotuner | Heuristics], Heuristics]:
    """Decorator computing derived constexpr kwargs from runtime arguments before grid evaluation."""

    def decorator(fn: JITFunction | Autotuner | Heuristics) -> Heuristics:
        return Heuristics(fn, values=values)

    return decorator


class _AutotuneGridLauncher:
    def __init__(self, tuner: Autotuner, grid: Any) -> None:
        self._tuner = tuner
        self._grid = grid

    def __call__(self, *args: Any, **kwargs: Any) -> CompiledKernel:
        return self._tuner._run(self._grid, args, kwargs)


class Autotuner:
    """Autotuning wrapper with roofline/PSO pruning, tensor restoration, and persistent disk caching."""

    def __init__(
        self,
        fn: JITFunction | Heuristics,
        configs: Sequence[Config],
        key: Sequence[str],
        *,
        warmup: int = 3,
        rep: int = 10,
        prune_configs_by: dict[str, Any] | None = None,
        reset_to_zero: Sequence[str] | None = None,
        restore_value: Sequence[str] | None = None,
        post_hook: Callable[[dict[str, Any], bool], None] | None = None,
        use_disk_cache: bool = True,
    ) -> None:
        if not isinstance(fn, (JITFunction, Heuristics)):
            raise TypeError("@autotune must decorate an @m1triton.jit or @m1triton.heuristics kernel")
        if not configs:
            raise ValueError("@autotune requires at least one Config")
        self.fn = fn
        self._jit_fn = _unwrap_jit_function(fn)
        self._param_names = self._jit_fn._param_names
        self.configs = list(configs)
        self.key = tuple(key)
        self.warmup = max(1, int(warmup))
        self.rep = max(1, int(rep))
        self.prune_configs_by = prune_configs_by or {}
        self.reset_to_zero = tuple(reset_to_zero or ())
        self.restore_value = tuple(restore_value or ())
        self.post_hook = post_hook
        self.use_disk_cache = bool(use_disk_cache)
        self.cache: dict[tuple[Any, ...], Config] = {}
        self.best_config: Config | None = None
        self.bench_count: int = 0
        self.disk_cache_hits: int = 0
        self._default_named: dict[str, Any] = {
            name: param.default
            for name, param in self._jit_fn._sig.parameters.items()
            if param.default is not inspect.Parameter.empty
        }
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
        named = dict(self._default_named)
        for name, val in zip(self._param_names, args):
            named[name] = val
        if kwargs:
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
            (name, v.dtype)
            for name in self._param_names
            if (v := named_args.get(name)) is not None and _is_buffer_arg(v)
        )
        return (key_vals, dtype_tags)

    def _build_disk_cache_identity(
        self,
        named_args: dict[str, Any],
        candidates: Sequence[Config],
    ) -> tuple[str, dict[str, Any]]:
        key_vals = [_canonicalize_json_value(named_args.get(k)) for k in self.key]
        dtype_tags = [
            [k, str(v.dtype)]
            for k, v in sorted(named_args.items())
            if _is_buffer_arg(v)
        ]
        candidate_sigs = [cfg.signature() for cfg in candidates]
        key_payload: dict[str, Any] = {
            "version": _DISK_CACHE_VERSION,
            "fn_name": self._jit_fn.__name__,
            "source_sha256": _get_fn_source_hash(self._jit_fn),
            "gpu_name": _get_gpu_name(),
            "key_names": list(self.key),
            "key_vals": key_vals,
            "dtype_tags": dtype_tags,
            "candidate_configs": candidate_sigs,
        }
        canonical = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return digest, key_payload

    def _disk_cache_enabled(self) -> bool:
        return self.use_disk_cache and not _is_disk_cache_disabled()

    def _load_disk_cache(
        self,
        cache_hash: str,
        candidates: Sequence[Config],
    ) -> Config | None:
        if not self._disk_cache_enabled():
            return None
        cache_path = get_autotune_cache_dir() / f"{cache_hash}.json"
        if not cache_path.is_file():
            return None
        try:
            with cache_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError, TypeError):
            return None

        if not isinstance(payload, dict):
            return None
        if payload.get("version") != _DISK_CACHE_VERSION:
            return None
        if payload.get("cache_hash") != cache_hash:
            return None
        saved_sig = payload.get("best_config")
        if not isinstance(saved_sig, dict):
            return None

        valid_pool = list(candidates) if candidates else list(self.configs)
        for cfg in valid_pool:
            if cfg.signature() == saved_sig:
                return cfg
        return None

    def _save_disk_cache(
        self,
        cache_hash: str,
        key_payload: dict[str, Any],
        chosen: Config,
    ) -> None:
        if not self._disk_cache_enabled():
            return
        cache_dir = get_autotune_cache_dir()
        cache_path = cache_dir / f"{cache_hash}.json"
        record = {
            "version": _DISK_CACHE_VERSION,
            "cache_hash": cache_hash,
            "key": key_payload,
            "best_config": chosen.signature(),
        }
        tmp_path: str | None = None
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".autotune_", suffix=".tmp", dir=str(cache_dir))
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh, sort_keys=True, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, cache_path)
        except OSError:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _snapshot_restore_buffers(self, named_args: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
        if not self.restore_value:
            return {}
        has_mps = any(_is_mps_tensor(named_args.get(name)) for name in self.restore_value)
        rt.synchronize()
        if has_mps:
            _sync_mps_if_loaded()
        saved: dict[str, tuple[Any, Any]] = {}
        for name in self.restore_value:
            buf = named_args.get(name)
            if buf is None:
                continue
            if hasattr(buf, "detach") and hasattr(buf, "clone") and hasattr(buf, "copy_"):
                saved[name] = (buf, buf.detach().clone())
            elif hasattr(buf, "copy") and hasattr(buf, "__setitem__"):
                saved[name] = (buf, buf.copy())
        if has_mps and saved:
            _sync_mps_if_loaded()
        return saved

    def _restore_buffers(self, saved: Mapping[str, tuple[Any, Any]]) -> None:
        if not saved:
            return
        rt.synchronize()
        restored_mps = False
        for buf, orig in saved.values():
            if hasattr(buf, "copy_"):
                buf.copy_(orig)
                if _is_mps_tensor(buf):
                    restored_mps = True
            else:
                buf[...] = orig
        if restored_mps:
            _sync_mps_if_loaded()

    def _check_pso_feasible(
        self,
        cfg: Config,
        args: Sequence[Any],
        base_kwargs: dict[str, Any],
    ) -> None:
        max_tg_fn = getattr(rt, "kernel_max_threads_per_threadgroup", None)
        if not callable(max_tg_fn):
            return
        compile_kwargs = dict(base_kwargs)
        compile_kwargs.update(cfg.all_kwargs())
        compiled = self.fn.compile(*args, **compile_kwargs)
        spec = compiled.launch_spec
        tg_threads = int(spec.tg_x) * int(spec.tg_y) * int(spec.tg_z)
        max_tg_threads = int(max_tg_fn(compiled.kernel_id))
        if max_tg_threads > 0 and tg_threads > max_tg_threads:
            raise _PSOInfeasibleError(
                f"Config {cfg!r} requires {tg_threads} threads/threadgroup, "
                f"exceeding compiled PSO limit {max_tg_threads}"
            )

    def _bench_config(
        self,
        cfg: Config,
        grid: Any,
        args: Sequence[Any],
        base_kwargs: dict[str, Any],
        named_args: dict[str, Any],
        saved_buffers: Mapping[str, tuple[Any, Any]] | None = None,
    ) -> float:
        self._check_pso_feasible(cfg, args, base_kwargs)

        cfg_kwargs = cfg.all_kwargs()
        merged_kwargs = dict(base_kwargs)
        merged_kwargs.update(cfg_kwargs)
        merged_kwargs["wait"] = True
        wrapped_grid = _wrap_grid_with_meta(grid, cfg_kwargs)

        def _prepare_trial() -> None:
            if saved_buffers:
                self._restore_buffers(saved_buffers)
            if cfg.pre_hook is not None:
                cfg.pre_hook(named_args)
            for name in self.reset_to_zero:
                buf = named_args.get(name)
                if buf is not None and hasattr(buf, "zero_"):
                    buf.zero_()

        self.bench_count += 1

        for _ in range(self.warmup):
            _prepare_trial()
            try:
                self.fn[wrapped_grid](*args, **merged_kwargs)
            except Exception:
                if self.post_hook is not None:
                    self.post_hook(named_args, True)
                raise
            if self.post_hook is not None:
                self.post_hook(named_args, False)

        times_ms: list[float] = []
        for _ in range(self.rep):
            _prepare_trial()
            t0 = time.perf_counter()
            try:
                compiled = self.fn[wrapped_grid](*args, **merged_kwargs)
            except Exception:
                if self.post_hook is not None:
                    self.post_hook(named_args, True)
                raise
            wall_ms = (time.perf_counter() - t0) * 1000.0
            if self.post_hook is not None:
                self.post_hook(named_args, False)
            gpu_ms = compiled.last_gpu_ms if compiled.last_gpu_ms > 0.0 else wall_ms
            times_ms.append(gpu_ms)
        return float(statistics.median(times_ms))

    def _select_config_with_status(
        self,
        grid: Any,
        args: Sequence[Any],
        kwargs: dict[str, Any],
        *,
        named_args: dict[str, Any] | None = None,
    ) -> tuple[Config, bool]:
        if named_args is None:
            named_args = self._named_args(args, kwargs)
        ckey = self._build_cache_key(named_args)
        cached = self.cache.get(ckey)
        if cached is not None:
            self.best_config = cached
            return cached, False

        elem_bytes = self._infer_elem_bytes(args, kwargs)
        candidates = roofline_filter_configs(self.configs, elem_bytes=elem_bytes)

        early_prune = self.prune_configs_by.get("early_config_prune")
        if callable(early_prune):
            pruned = early_prune(candidates, named_args)
            if pruned:
                candidates = list(pruned)

        disk_hash, disk_key_payload = self._build_disk_cache_identity(named_args, candidates)
        disk_cached = self._load_disk_cache(disk_hash, candidates)
        if disk_cached is not None:
            self.cache[ckey] = disk_cached
            self.best_config = disk_cached
            self.disk_cache_hits += 1
            return disk_cached, False

        ran_tuning = False
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            saved_buffers = self._snapshot_restore_buffers(named_args)
            best_cfg: Config | None = None
            best_ms = float("inf")
            last_err: Exception | None = None
            ran_tuning = True
            try:
                for cfg in candidates:
                    try:
                        med_ms = self._bench_config(
                            cfg,
                            grid,
                            args,
                            kwargs,
                            named_args,
                            saved_buffers=saved_buffers,
                        )
                    except (_PSOInfeasibleError, RuntimeError, Exception) as exc:
                        last_err = exc
                        continue
                    if med_ms < best_ms:
                        best_ms = med_ms
                        best_cfg = cfg
            finally:
                self._restore_buffers(saved_buffers)
            if best_cfg is None:
                raise RuntimeError(f"All autotune configs failed for '{self.__name__}'") from last_err
            chosen = best_cfg

        self.cache[ckey] = chosen
        self.best_config = chosen
        self._save_disk_cache(disk_hash, disk_key_payload, chosen)
        return chosen, ran_tuning

    def _select_config(
        self,
        grid: Any,
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> Config:
        chosen, _ = self._select_config_with_status(grid, args, kwargs)
        return chosen

    def _run(self, grid: Any, args: Sequence[Any], kwargs: dict[str, Any]) -> CompiledKernel:
        named_args = self._named_args(args, kwargs)
        cfg, ran_tuning = self._select_config_with_status(grid, args, kwargs, named_args=named_args)
        if ran_tuning:
            if cfg.pre_hook is not None:
                cfg.pre_hook(named_args)
            for name in self.reset_to_zero:
                buf = named_args.get(name)
                if buf is not None and hasattr(buf, "zero_"):
                    buf.zero_()
        cfg_kwargs = cfg.all_kwargs()
        merged = dict(kwargs)
        merged.update(cfg_kwargs)
        wrapped_grid = _wrap_grid_with_meta(grid, cfg_kwargs)
        return self.fn[wrapped_grid](*args, **merged)

    def compile(self, *args: Any, **kwargs: Any) -> CompiledKernel:
        cfg = self.best_config or roofline_filter_configs(self.configs)[0]
        merged = dict(kwargs)
        merged.update(cfg.all_kwargs())
        return self.fn.compile(*args, **merged)

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
    restore_value: Sequence[str] | None = None,
    post_hook: Callable[[dict[str, Any], bool], None] | None = None,
    use_disk_cache: bool = True,
) -> Callable[[JITFunction | Heuristics], Autotuner]:
    """Decorator attaching roofline-pruned GPU autotuning over an `@m1triton.jit` or `@heuristics` kernel."""

    def decorator(fn: JITFunction | Heuristics) -> Autotuner:
        return Autotuner(
            fn,
            configs=configs,
            key=key,
            warmup=warmup,
            rep=rep,
            prune_configs_by=prune_configs_by,
            reset_to_zero=reset_to_zero,
            restore_value=restore_value,
            post_hook=post_hook,
            use_disk_cache=use_disk_cache,
        )

    return decorator


__all__ = [
    "Autotuner",
    "Config",
    "Heuristics",
    "autotune",
    "get_autotune_cache_dir",
    "heuristics",
    "roofline_filter_configs",
]
