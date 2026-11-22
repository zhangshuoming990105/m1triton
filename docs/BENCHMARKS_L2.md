# M1Triton KernelBench Level 2 Scoreboard (`v2.0`)

- KernelBench Commit: `423217d9fda9`
- Harness: SHA-256-locked [evals/eval_kernelbench_l2.py](../evals/eval_kernelbench_l2.py) (`--seeds 3 --warmup 10 --trials 30 --process-isolated --chunk-size 5`)
- Passed: **100/100** | Beats Baseline (`>1.00x`): **100/100** | Geomean Speedup: **3.0130x** | Median Dispatches/Forward: **2**
- Raw Artifact: [.audit/eval_final_l2_100.json](../.audit/eval_final_l2_100.json)

## Category Breakdown (`[measured]`)

| Fusion Family | Tasks | Passed | Beats MPS (`>1.00x`) | Median DPF | Geomean Speedup | Top Speedups |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GEMM / Matmul / BMM Fusion | 37 | 37/37 | 37/37 | 2 | **1.69x** | `#37` **2.15x**, `#51` **1.94x**, `#66` **1.92x** |
| Conv2d Fusion | 21 | 21/21 | 21/21 | 3 | **2.09x** | `#82` **4.63x**, `#57` **4.45x**, `#21` **4.31x** |
| Conv3d Fusion | 12 | 12/12 | 12/12 | 3 | **3.98x** | `#27` **22.04x**, `#90` **6.47x**, `#48` **5.90x** |
| ConvTranspose2d Fusion | 11 | 11/11 | 11/11 | 2 | **3.28x** | `#42` **32.57x**, `#2` **5.39x**, `#16` **4.21x** |
| ConvTranspose3d Fusion | 19 | 19/19 | 19/19 | 2 | **11.13x** | `#77` **421.37x**, `#78` **20.39x**, `#34` **16.37x** |
| **Total** | **100** | **100/100** | **100/100** | **2** | **3.0130x** | Zero PyTorch/MPSGraph/Inductor fallbacks |

## Complete 100-Task Scoreboard (`[measured]`)

| ID | Task | Status | Base MPS (ms) | M1Triton (ms) | Speedup | DPF | Max Abs Err |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `Conv2D_ReLU_BiasAdd` | passed | 60.5923 | 36.2192 | 1.67x | 2 | 4.77e-07 |
| 2 | `ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide` | passed | 259.9686 | 48.1983 | 5.39x | 1 | 1.07e-06 |
| 3 | `ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU` | passed | 396.1816 | 54.2859 | 7.30x | 2 | 5.48e-06 |
| 4 | `Conv2d_Mish_Mish` | passed | 93.9557 | 60.7813 | 1.55x | 112 | 7.51e-06 |
| 5 | `ConvTranspose2d_Subtract_Tanh` | passed | 304.9599 | 104.5586 | 2.92x | 1 | 9.56e-07 |
| 6 | `Conv3d_Softmax_MaxPool_MaxPool` | passed | 12.4699 | 3.6134 | 3.45x | 3 | 8.94e-08 |
| 7 | `Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd` | passed | 146.7175 | 28.9843 | 5.06x | 1 | 4.77e-07 |
| 8 | `Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum` | passed | 99.5873 | 18.1176 | 5.50x | 3 | 7.15e-07 |
| 9 | `Matmul_Subtract_Multiply_ReLU` | passed | 48.0075 | 26.5654 | 1.81x | 1 | 0.00e+00 |
| 10 | `ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh` | passed | 207.8851 | 93.7608 | 2.22x | 3 | 1.79e-07 |
| 11 | `ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm` | passed | 78.5938 | 71.8990 | 1.09x | 5 | 1.20e-05 |
| 12 | `Gemm_Multiply_LeakyReLU` | passed | 41.5936 | 24.6559 | 1.69x | 1 | 0.00e+00 |
| 13 | `ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling` | passed | 556.4964 | 117.0189 | 4.76x | 3 | 2.38e-07 |
| 14 | `Gemm_Divide_Sum_Scaling` | passed | 41.0561 | 25.7659 | 1.59x | 2 | 2.20e-03 |
| 15 | `ConvTranspose3d_BatchNorm_Subtract` | passed | 56.1732 | 7.9052 | 7.11x | 4 | 3.87e-07 |
| 16 | `ConvTranspose2d_Mish_Add_Hardtanh_Scaling` | passed | 264.3786 | 62.8460 | 4.21x | 1 | 1.91e-06 |
| 17 | `Conv2d_InstanceNorm_Divide` | passed | 44.8579 | 39.5255 | 1.13x | 60 | 3.03e-05 |
| 18 | `Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp` | passed | 48.0285 | 28.6777 | 1.67x | 2 | 1.91e-05 |
| 19 | `ConvTranspose2d_GELU_GroupNorm` | passed | 184.3125 | 129.9599 | 1.42x | 4 | 3.81e-06 |
| 20 | `ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd` | passed | 222.2757 | 16.5630 | 13.42x | 1 | 1.67e-06 |
| 21 | `Conv2d_Add_Scale_Sigmoid_GroupNorm` | passed | 79.8170 | 18.5117 | 4.31x | 4 | 3.05e-05 |
| 22 | `Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish` | passed | 43.0872 | 24.6212 | 1.75x | 2 | 3.81e-05 |
| 23 | `Conv3d_GroupNorm_Mean` | passed | 15.0899 | 5.9576 | 2.53x | 3 | 6.61e-08 |
| 24 | `Conv3d_Min_Softmax` | passed | 15.8632 | 6.0349 | 2.63x | 3 | 3.73e-08 |
| 25 | `Conv2d_Min_Tanh_Tanh` | passed | 33.1166 | 28.6364 | 1.16x | 3 | 3.58e-07 |
| 26 | `ConvTranspose3d_Add_HardSwish` | passed | 426.8295 | 38.9659 | 10.95x | 1 | 9.54e-07 |
| 27 | `Conv3d_HardSwish_GroupNorm_Mean` | passed | 625.9117 | 28.3940 | 22.04x | 3 | 5.36e-07 |
| 28 | `BMM_InstanceNorm_Sum_ResidualAdd_Multiply` | passed | 43.9568 | 27.1905 | 1.62x | 3 | 9.54e-07 |
| 29 | `Matmul_Mish_Mish` | passed | 43.8988 | 26.0969 | 1.68x | 1 | 3.58e-07 |
| 30 | `Gemm_GroupNorm_Hardtanh` | passed | 44.1054 | 27.3028 | 1.62x | 3 | 5.96e-07 |
| 31 | `Conv2d_Min_Add_Multiply` | passed | 65.8804 | 36.1719 | 1.82x | 2 | 9.54e-07 |
| 32 | `Conv2d_Scaling_Min` | passed | 80.1300 | 69.5123 | 1.15x | 113 | 1.53e-05 |
| 33 | `Gemm_Scale_BatchNorm` | passed | 42.4349 | 25.4863 | 1.67x | 2 | 2.38e-07 |
| 34 | `ConvTranspose3d_LayerNorm_GELU_Scaling` | passed | 935.9640 | 57.1738 | 16.37x | 2 | 1.12e-05 |
| 35 | `Conv2d_Subtract_HardSwish_MaxPool_Mish` | passed | 68.5629 | 43.8674 | 1.56x | 3 | 3.58e-07 |
| 36 | `ConvTranspose2d_Min_Sum_GELU_Add` | passed | 43.3704 | 13.8082 | 3.14x | 3 | 0.00e+00 |
| 37 | `Matmul_Swish_Sum_GroupNorm` | passed | 92.8025 | 43.0794 | 2.15x | 2 | 1.19e-06 |
| 38 | `ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply` | passed | 464.5610 | 51.1860 | 9.08x | 4 | 5.64e-11 |
| 39 | `Gemm_Scale_BatchNorm` | passed | 116.9634 | 69.1672 | 1.69x | 2 | 4.77e-07 |
| 40 | `Matmul_Scaling_ResidualAdd` | passed | 109.8614 | 67.7563 | 1.62x | 1 | 0.00e+00 |
| 41 | `Gemm_BatchNorm_GELU_ReLU` | passed | 118.9661 | 85.4679 | 1.39x | 2 | 2.38e-07 |
| 42 | `ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply` | passed | 119.8102 | 3.6783 | 32.57x | 4 | 7.63e-06 |
| 43 | `Conv3d_Max_LogSumExp_ReLU` | passed | 82.6870 | 81.0204 | 1.02x | 3 | 1.43e-06 |
| 44 | `ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean` | passed | 44.8391 | 13.8885 | 3.23x | 2 | 1.49e-08 |
| 45 | `Gemm_Sigmoid_LogSumExp` | passed | 61.5105 | 52.3599 | 1.17x | 3 | 1.43e-06 |
| 46 | `Conv2d_Subtract_Tanh_Subtract_AvgPool` | passed | 48.1741 | 34.2984 | 1.40x | 59 | 2.53e-06 |
| 47 | `Conv3d_Mish_Tanh` | passed | 79.2389 | 39.9738 | 1.98x | 1 | 2.03e-06 |
| 48 | `Conv3d_Scaling_Tanh_Multiply_Sigmoid` | passed | 56.6888 | 9.6050 | 5.90x | 1 | 2.98e-07 |
| 49 | `ConvTranspose3d_Softmax_Sigmoid` | passed | 205.7838 | 21.2249 | 9.70x | 2 | 1.19e-07 |
| 50 | `ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling` | passed | 180.8505 | 20.5968 | 8.78x | 2 | 2.38e-07 |
| 51 | `Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd` | passed | 88.0399 | 45.3682 | 1.94x | 3 | 1.19e-07 |
| 52 | `Conv2d_Activation_BatchNorm` | passed | 28.4751 | 19.7069 | 1.44x | 3 | 5.96e-07 |
| 53 | `Gemm_Scaling_Hardtanh_GELU` | passed | 85.4126 | 44.7518 | 1.91x | 1 | 1.79e-07 |
| 54 | `Conv2d_Multiply_LeakyReLU_GELU` | passed | 67.1297 | 38.1822 | 1.76x | 2 | 9.54e-07 |
| 55 | `Matmul_MaxPool_Sum_Scale` | passed | 252.5772 | 153.0294 | 1.65x | 2 | 2.44e-04 |
| 56 | `Matmul_Sigmoid_Sum` | passed | 254.2801 | 157.0502 | 1.62x | 2 | 3.91e-03 |
| 57 | `Conv2d_ReLU_HardSwish` | passed | 18.4989 | 4.1593 | 4.45x | 2 | 4.77e-07 |
| 58 | `ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp` | passed | 188.6191 | 18.8428 | 10.01x | 2 | 2.38e-07 |
| 59 | `Matmul_Swish_Scaling` | passed | 254.3693 | 154.0228 | 1.65x | 1 | 4.77e-07 |
| 60 | `ConvTranspose3d_Swish_GroupNorm_HardSwish` | passed | 198.2599 | 65.1589 | 3.04x | 3 | 3.34e-06 |
| 61 | `ConvTranspose3d_ReLU_GroupNorm` | passed | 313.5138 | 58.5487 | 5.35x | 3 | 2.15e-05 |
| 62 | `Matmul_GroupNorm_LeakyReLU_Sum` | passed | 45.5174 | 26.7404 | 1.70x | 2 | 1.91e-06 |
| 63 | `Gemm_ReLU_Divide` | passed | 44.5632 | 26.6930 | 1.67x | 1 | 0.00e+00 |
| 64 | `Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU` | passed | 45.4664 | 26.3231 | 1.73x | 2 | 1.91e-06 |
| 65 | `Conv2d_AvgPool_Sigmoid_Sum` | passed | 44.2944 | 12.4069 | 3.57x | 3 | 6.25e-02 |
| 66 | `Matmul_Dropout_Softmax` | passed | 35.8888 | 18.6759 | 1.92x | 2 | 1.16e-10 |
| 67 | `Conv2d_GELU_GlobalAvgPool` | passed | 30.1845 | 21.7758 | 1.39x | 3 | 1.19e-07 |
| 68 | `Matmul_Min_Subtract` | passed | 36.5558 | 19.2654 | 1.90x | 1 | 2.38e-07 |
| 69 | `Conv2d_HardSwish_ReLU` | passed | 9.1029 | 4.2389 | 2.15x | 2 | 4.17e-07 |
| 70 | `Gemm_Sigmoid_Scaling_ResidualAdd` | passed | 44.2351 | 25.9619 | 1.70x | 1 | 4.77e-07 |
| 71 | `Conv2d_Divide_LeakyReLU` | passed | 9.0882 | 4.2709 | 2.13x | 2 | 1.79e-07 |
| 72 | `ConvTranspose3d_BatchNorm_AvgPool_AvgPool` | passed | 184.4605 | 21.2671 | 8.67x | 3 | 2.98e-08 |
| 73 | `Conv2d_BatchNorm_Scaling` | passed | 9.2546 | 4.2934 | 2.16x | 3 | 9.54e-07 |
| 74 | `ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max` | passed | 63.1487 | 6.0636 | 10.41x | 2 | 4.77e-07 |
| 75 | `Gemm_GroupNorm_Min_BiasAdd` | passed | 46.4605 | 27.4866 | 1.69x | 3 | 9.54e-07 |
| 76 | `Gemm_Add_ReLU` | passed | 44.7020 | 26.5173 | 1.69x | 1 | 0.00e+00 |
| 77 | `ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool` | passed | 967.4520 | 2.2960 | 421.37x | 4 | 2.98e-07 |
| 78 | `ConvTranspose3d_Max_Max_Sum` | passed | 2010.5680 | 98.6233 | 20.39x | 3 | 7.63e-06 |
| 79 | `Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max` | passed | 17.2407 | 3.5935 | 4.80x | 3 | 4.23e-06 |
| 80 | `Gemm_Max_Subtract_GELU` | passed | 44.4900 | 27.4702 | 1.62x | 2 | 0.00e+00 |
| 81 | `Gemm_Swish_Divide_Clamp_Tanh_Clamp` | passed | 45.3600 | 26.2055 | 1.73x | 1 | 1.79e-07 |
| 82 | `Conv2d_Tanh_Scaling_BiasAdd_Max` | passed | 124.3754 | 26.8817 | 4.63x | 3 | 7.15e-07 |
| 83 | `Conv3d_GroupNorm_Min_Clamp_Dropout` | passed | 39.9644 | 13.1539 | 3.04x | 3 | 0.00e+00 |
| 84 | `Gemm_BatchNorm_Scaling_Softmax` | passed | 45.5846 | 28.7798 | 1.58x | 3 | 2.33e-10 |
| 85 | `Conv2d_GroupNorm_Scale_MaxPool_Clamp` | passed | 26.4810 | 8.4368 | 3.14x | 4 | 1.31e-06 |
| 86 | `Matmul_Divide_GELU` | passed | 44.4461 | 27.4247 | 1.62x | 1 | 3.73e-08 |
| 87 | `Conv2d_Subtract_Subtract_Mish` | passed | 54.6821 | 16.9398 | 3.23x | 2 | 5.36e-07 |
| 88 | `Gemm_GroupNorm_Swish_Multiply_Swish` | passed | 46.8910 | 27.9984 | 1.67x | 2 | 2.86e-06 |
| 89 | `ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max` | passed | 198.8661 | 21.5776 | 9.22x | 3 | 7.15e-07 |
| 90 | `Conv3d_LeakyReLU_Sum_Clamp_GELU` | passed | 349.7740 | 54.0472 | 6.47x | 1 | 1.67e-06 |
| 91 | `ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid` | passed | 186.7224 | 73.2326 | 2.55x | 2 | 1.79e-07 |
| 92 | `Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp` | passed | 27.8810 | 9.3018 | 3.00x | 4 | 1.43e-06 |
| 93 | `ConvTranspose2d_Add_Min_GELU_Multiply` | passed | 151.2805 | 62.2485 | 2.43x | 1 | 5.95e-07 |
| 94 | `Gemm_BiasAdd_Hardtanh_Mish_GroupNorm` | passed | 53.6398 | 31.0298 | 1.73x | 2 | 1.07e-06 |
| 95 | `Matmul_Add_Swish_Tanh_GELU_Hardtanh` | passed | 51.3487 | 29.1746 | 1.76x | 1 | 4.17e-07 |
| 96 | `ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp` | passed | 185.2798 | 19.3896 | 9.56x | 2 | 2.24e-08 |
| 97 | `Matmul_BatchNorm_BiasAdd_Divide_Swish` | passed | 51.8303 | 32.3260 | 1.60x | 2 | 3.58e-07 |
| 98 | `Matmul_AvgPool_GELU_Scale_Max` | passed | 50.1379 | 28.1401 | 1.78x | 2 | 1.79e-07 |
| 99 | `Matmul_GELU_Softmax` | passed | 51.6745 | 28.7388 | 1.80x | 2 | 2.33e-10 |
| 100 | `ConvTranspose3d_Clamp_Min_Divide` | passed | 2612.8829 | 174.7249 | 14.95x | 1 | 3.58e-07 |
