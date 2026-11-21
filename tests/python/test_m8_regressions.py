"""Regression tests for M8 adversarial audit findings (Winograd PSO limits, activation NaNs, cross-queue sync)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

import m1triton.runtime as m1_runtime

_KERNELS_DIR = str(Path(__file__).resolve().parents[2] / "benchmarks" / "kernels")
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

import _activations_common as act
import _conv_std_common as conv_std
import _loss_common as loss_common
import _scan_common as scan_common
import task_089
import task_090

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")


@pytest.mark.parametrize(
    ("c_in", "c_out", "h", "w"),
    [
        (512, 512, 28, 28),
        (256, 256, 56, 56),
        (128, 128, 112, 112),
        (64, 64, 64, 56),
        (64, 64, 56, 64),
        (64, 64, 96, 96),
    ],
)
def test_winograd_non_pow2_widths_and_unscaled_accuracy(c_in: int, c_out: int, h: int, w: int) -> None:
    torch.manual_seed(0)
    x = torch.randn((1, c_in, h, w), device="mps", dtype=torch.float32)
    weight = torch.randn((c_out, c_in, 3, 3), device="mps", dtype=torch.float32) * (1.0 / ((c_in * 9) ** 0.5))
    ref = torch.nn.functional.conv2d(x, weight, padding=1)
    cand = conv_std.conv_nd(x, weight, None, ndim=2, padding=1)
    m1_runtime.synchronize()
    assert torch.allclose(cand, ref, rtol=1e-4, atol=1e-4), f"max_err={(cand - ref).abs().max().item()}"


def test_metal_runtime_rejects_threadgroup_exceeding_pso_limit() -> None:
    x = torch.randn((1, 64, 64, 56), device="mps", dtype=torch.float32)
    w = torch.randn((64, 64, 3, 3), device="mps", dtype=torch.float32) * 0.05
    with pytest.raises(RuntimeError, match="maxTotalThreadsPerThreadgroup"):
        conv_std.conv2d_winograd_3x3(
            x,
            w,
            None,
            padding=(1, 1),
            wino_tile=(1, 32, 32, 16, 2, 16, 16, 1),
            use_staged_4x3=False,
        )


@pytest.mark.parametrize("n", [4096, 4097])
def test_activations_propagate_nans(n: int) -> None:
    torch.manual_seed(7)
    x_cpu = torch.randn((n,), dtype=torch.float32)
    nan_idx = torch.arange(0, n, 3)
    x_cpu[nan_idx] = float("nan")
    x = x_cpu.to("mps")

    cases = [
        ("relu", act.relu(x), torch.relu(x)),
        ("leaky_relu", act.leaky_relu(x, 0.01), torch.nn.functional.leaky_relu(x, 0.01)),
        ("selu", act.selu(x), torch.nn.functional.selu(x)),
        ("elu", act.elu(x, 1.0), torch.nn.functional.elu(x, 1.0)),
        ("hardtanh", act.hardtanh(x), torch.nn.functional.hardtanh(x)),
        ("hardsigmoid", act.hardsigmoid(x), torch.nn.functional.hardsigmoid(x)),
    ]
    m1_runtime.synchronize()
    for name, cand, ref in cases:
        assert torch.equal(torch.isnan(cand), torch.isnan(ref)), f"{name} lost NaNs for n={n}"
        finite = ~torch.isnan(ref)
        assert torch.allclose(cand[finite], ref[finite], rtol=1e-4, atol=1e-4), f"{name} mismatch for n={n}"


def test_cross_queue_sync_losses_and_non_last_dim_scans() -> None:
    torch.manual_seed(11)
    for b in (1, 3, 2):
        logits = torch.randn((b, 4099), device="mps", dtype=torch.float32)
        p = torch.softmax(logits, dim=-1)
        q = torch.softmax(torch.randn((b, 4099), device="mps", dtype=torch.float32), dim=-1)
        got = loss_common.kl_div_loss(p, q).cpu().item()
        expected = torch.nn.functional.kl_div(torch.log(p), q, reduction="batchmean").cpu().item()
        assert abs(got - expected) < 1e-4, f"kl_div cross-queue mismatch: got={got} expected={expected}"

    x = torch.randn((4096, 8), device="mps", dtype=torch.float32)
    out_sum = task_089.ModelNew(dim=0)(x)
    ref_sum = torch.cumsum(x, dim=0)
    assert torch.allclose(out_sum, ref_sum, rtol=1e-3, atol=1e-3)

    x_small = torch.ones((4096, 8), device="mps", dtype=torch.float32) * 1.0001
    out_prod = task_090.ModelNew(dim=0)(x_small)
    ref_prod = torch.cumprod(x_small, dim=0)
    assert torch.allclose(out_prod, ref_prod, rtol=1e-3, atol=1e-3)

    x_f16 = torch.randn((16, 64), device="mps", dtype=torch.float16)
    out_f16 = scan_common.row_scan(x_f16, dim=0, exclusive=True)
    ref_f16 = torch.cumsum(x_f16.float(), dim=0) - x_f16.float()
    assert out_f16.dtype == torch.float16
    assert torch.allclose(out_f16.float(), ref_f16, rtol=1e-2, atol=1e-2)
