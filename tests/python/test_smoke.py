import m1triton.runtime as rt


def test_target_spec_and_metal_device():
    spec = rt.target_spec()
    assert spec.simd_width == 32
    assert spec.max_threadgroup_memory_bytes == 32768

    info = rt.device_info()
    assert info.available is True
    assert info.has_unified_memory is True
    assert info.max_threadgroup_memory_bytes >= 32768

    rt.reset_dispatch_count()
    assert rt.get_dispatch_count() == 0

    ok = rt.compile_msl_smoke(
        """
        #include <metal_stdlib>
        using namespace metal;
        kernel void noop_kernel(device float* x [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
            x[gid] = x[gid];
        }
        """,
        "noop_kernel",
    )
    assert ok is True
