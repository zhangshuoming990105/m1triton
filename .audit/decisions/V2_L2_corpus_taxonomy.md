# KernelBench Level 2 Corpus Taxonomy (100 Tasks)

- Upstream Commit: `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`

| ID | Producer | Name | Key Shapes / Hints |
| --- | --- | --- | --- |
| 1 | `Conv2d` | `Conv2D_ReLU_BiasAdd` | `batch_size = 128, in_channels  = 64, out_channels = 128, height = width = 128, kernel_size = 3` |
| 2 | `ConvTranspose2d` | `ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide` | `batch_size = 128, in_channels  = 64, out_channels = 64, height = width = 128, kernel_size = 3, stride = 2` |
| 3 | `ConvTranspose3d` | `ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU` | `batch_size = 32, in_channels = 32, out_channels = 64, depth, height, width = 16, 32, 32, kernel_size = (3, 3, 3), stride = (2, 2, 2)` |
| 4 | `Conv2d` | `Conv2d_Mish_Mish` | `batch_size   = 64, in_channels  = 64, out_channels = 128, height = width = 256, kernel_size = 3` |
| 5 | `ConvTranspose2d` | `ConvTranspose2d_Subtract_Tanh` | `batch_size = 32, in_channels  = 64, out_channels = 64, height = width = 256, kernel_size = 4` |
| 6 | `Conv3d` | `Conv3d_Softmax_MaxPool_MaxPool` | `batch_size = 128, in_channels = 3, out_channels = 16, depth, height, width = 16, 32, 32, kernel_size = 3` |
| 7 | `Conv3d` | `Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd` | `batch_size = 64, in_channels = 8, out_channels = 32, depth, height, width = 32, 64, 64, kernel_size = 3` |
| 8 | `Conv3d` | `Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum` | `batch_size   = 128, in_channels  = 8, out_channels = 16, depth = 16; height = width = 64, kernel_size = (3, 3, 3)` |
| 9 | `GEMM/Linear` | `Matmul_Subtract_Multiply_ReLU` | `batch_size = 1024, in_features = 8192, out_features = 8192` |
| 10 | `ConvTranspose2d` | `ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh` | `batch_size = 128, in_channels  = 64, out_channels = 64, height = width = 256, kernel_size  = 3, stride = 1` |
| 11 | `ConvTranspose2d` | `ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm` | `batch_size = 512, in_channels  = 64, out_channels = 128, height = width = 2048, kernel_size  = 5, stride       = 1` |
| 12 | `GEMM/Linear` | `Gemm_Multiply_LeakyReLU` | `batch_size = 1024, in_features  = 8192, out_features = 8192` |
| 13 | `ConvTranspose3d` | `ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling` | `batch_size = 16, in_channels  = 16, out_channels = 64, depth = 32; height = width = 128, kernel_size  = 3, stride       = 1` |
| 14 | `GEMM/Linear` | `Gemm_Divide_Sum_Scaling` | `batch_size   = 1024, input_size   = 8192, hidden_size  = 8192` |
| 15 | `ConvTranspose3d` | `ConvTranspose3d_BatchNorm_Subtract` | `batch_size = 16, in_channels = 16, out_channels = 32, depth, height, width = 16, 32, 32, kernel_size = 3, stride = 2` |
| 16 | `ConvTranspose2d` | `ConvTranspose2d_Mish_Add_Hardtanh_Scaling` | `batch_size = 128, in_channels  = 64, out_channels = 64, height = width = 128, kernel_size  = 3, stride       = 2` |
| 17 | `Conv2d` | `Conv2d_InstanceNorm_Divide` | `batch_size = 128, in_channels  = 64, out_channels = 128, height = width = 128, kernel_size = 3` |
| 18 | `GEMM/Linear` | `Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp` | `batch_size = 1024, in_features  = 8192, out_features = 8192` |
| 19 | `ConvTranspose2d` | `ConvTranspose2d_GELU_GroupNorm` | `batch_size   = 128, in_channels  = 64, out_channels = 64, height = width = 256, kernel_size  = 3, stride       = 1` |
| 20 | `ConvTranspose3d` | `ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd` | `batch_size = 16, in_channels = 32, out_channels = 64, depth, height, width = 16, 32, 32, kernel_size = 3, stride = 2` |
| 21 | `Conv2d` | `Conv2d_Add_Scale_Sigmoid_GroupNorm` | `batch_size = 128, in_channels = 8, out_channels = 32, height = width = 256, kernel_size = 3, num_groups = 8` |
| 22 | `GEMM/Linear` | `Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish` | `batch_size = 1024, input_size = 8192, hidden_size = 8192` |
| 23 | `Conv3d` | `Conv3d_GroupNorm_Mean` | `batch_size = 128, in_channels = 3, out_channels = 24, kernel_size = 3, num_groups = 8` |
| 24 | `Conv3d` | `Conv3d_Min_Softmax` | `batch_size = 128, in_channels = 3, out_channels = 24, kernel_size = 3` |
| 25 | `Conv2d` | `Conv2d_Min_Tanh_Tanh` | `batch_size = 128, in_channels = 16, out_channels = 64, height = width = 256, kernel_size = 3` |
| 26 | `ConvTranspose3d` | `ConvTranspose3d_Add_HardSwish` | `batch_size = 128, in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1` |
| 27 | `Conv3d` | `Conv3d_HardSwish_GroupNorm_Mean` | `batch_size = 1024, in_channels = 3, out_channels = 16, depth, height, width = 16, 32, 32, kernel_size = 4` |
| 28 | `GEMM/Linear` | `BMM_InstanceNorm_Sum_ResidualAdd_Multiply` | `batch_size = 1024, in_features = 8192, out_features = 8192` |
| 29 | `GEMM/Linear` | `Matmul_Mish_Mish` | `batch_size = 1024, in_features = 8192, out_features = 8192` |
| 30 | `GEMM/Linear` | `Gemm_GroupNorm_Hardtanh` | `batch_size = 1024, in_features = 8192, out_features = 8192, num_groups = 16` |
| 31 | `Conv2d` | `Conv2d_Min_Add_Multiply` | `batch_size = 128, in_channels = 64, out_channels = 128, height = width = 128, kernel_size = 3` |
| 32 | `Conv2d` | `Conv2d_Scaling_Min` | `batch_size = 64, in_channels = 64, out_channels = 128, height = width = 256, kernel_size = 3` |
| 33 | `GEMM/Linear` | `Gemm_Scale_BatchNorm` | `batch_size = 1024, in_features = 8192, out_features = 8192` |
| 34 | `ConvTranspose3d` | `ConvTranspose3d_LayerNorm_GELU_Scaling` | `batch_size = 32, in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 1` |
| 35 | `Conv2d` | `Conv2d_Subtract_HardSwish_MaxPool_Mish` | `batch_size = 128, in_channels = 64, out_channels = 128, height = width = 128, kernel_size = 3` |
| 36 | `ConvTranspose2d` | `ConvTranspose2d_Min_Sum_GELU_Add` | `batch_size = 16, in_channels = 64, out_channels = 128, height, width = 128, 128, kernel_size = 3, stride = 2` |
| 37 | `GEMM/Linear` | `Matmul_Swish_Sum_GroupNorm` | `batch_size = 32768, in_features = 1024, out_features = 4096, num_groups = 64` |
| 38 | `ConvTranspose3d` | `ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply` | `batch_size = 32, in_channels = 32, out_channels = 64, depth, height, width = 32, 64, 64, kernel_size = 3, stride = 2` |
| 39 | `GEMM/Linear` | `Gemm_Scale_BatchNorm` | `batch_size = 16384, in_features = 4096, out_features = 4096` |
| 40 | `GEMM/Linear` | `Matmul_Scaling_ResidualAdd` | `in_features (int): Number of input features., out_features (int): Number of output features., batch_size = 16384, in_features = 4096, out_features = 4096` |
| 41 | `GEMM/Linear` | `Gemm_BatchNorm_GELU_ReLU` | `batch_size = 16384, in_features = 4096, out_features = 4096` |
| 42 | `ConvTranspose2d` | `ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply` | `batch_size = 16, in_channels = 64, out_channels = 128, height = width = 512, kernel_size = 3` |
| 43 | `Conv3d` | `Conv3d_Max_LogSumExp_ReLU` | `batch_size = 4, in_channels = 32, out_channels = 64, depth, height, width = 32, 128, 128, kernel_size = 3, stride = 1` |
| 44 | `ConvTranspose2d` | `ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean` | `batch_size = 16, in_channels = 64, out_channels = 128, height, width = 128, 128, kernel_size = 3, stride = 2` |
| 45 | `GEMM/Linear` | `Gemm_Sigmoid_LogSumExp` | `batch_size = 16384, input_size = 2048, hidden_size = 4096` |
| 46 | `Conv2d` | `Conv2d_Subtract_Tanh_Subtract_AvgPool` | `batch_size = 128, in_channels = 64, out_channels = 128, height, width = 128, 128, kernel_size = 3, kernel_size_pool = 2` |
| 47 | `Conv3d` | `Conv3d_Mish_Tanh` | `batch_size = 16, in_channels = 32, out_channels = 64, kernel_size = 3` |
| 48 | `Conv3d` | `Conv3d_Scaling_Tanh_Multiply_Sigmoid` | `batch_size = 128, in_channels = 3, out_channels = 16, depth, height, width = 16, 64, 64, kernel_size = 3` |
| 49 | `ConvTranspose3d` | `ConvTranspose3d_Softmax_Sigmoid` | `batch_size = 16, in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1` |
| 50 | `ConvTranspose3d` | `ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling` | `batch_size = 128, in_channels = 3, out_channels = 16, depth, height, width = 16, 32, 32, kernel_size = 3, stride = 2` |
| 51 | `GEMM/Linear` | `Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd` | `batch_size = 2048, in_features = 8192, out_features = 8192` |
| 52 | `Conv2d` | `Conv2d_Activation_BatchNorm` | `batch_size = 64, in_channels = 64, out_channels = 128, height, width = 128, 128, kernel_size = 3` |
| 53 | `GEMM/Linear` | `Gemm_Scaling_Hardtanh_GELU` | `batch_size = 2048, in_features = 8192, out_features = 8192` |
| 54 | `Conv2d` | `Conv2d_Multiply_LeakyReLU_GELU` | `batch_size = 64, in_channels = 64, out_channels = 64, height, width = 256, 256, kernel_size = 3` |
| 55 | `GEMM/Linear` | `Matmul_MaxPool_Sum_Scale` | `batch_size = 128, in_features = 32768, out_features = 32768, kernel_size = 2` |
| 56 | `GEMM/Linear` | `Matmul_Sigmoid_Sum` | `batch_size = 128, input_size = 32768, hidden_size = 32768` |
| 57 | `Conv2d` | `Conv2d_ReLU_HardSwish` | `batch_size = 128, in_channels = 8, out_channels = 64, height, width = 128, 128, kernel_size = 3` |
| 58 | `ConvTranspose3d` | `ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp` | `batch_size = 128, in_channels = 3, out_channels = 16, depth, height, width = 16, 32, 32, kernel_size = 3, stride = 2` |
| 59 | `GEMM/Linear` | `Matmul_Swish_Scaling` | `batch_size = 128, in_features = 32768, out_features = 32768` |
| 60 | `ConvTranspose3d` | `ConvTranspose3d_Swish_GroupNorm_HardSwish` | `batch_size = 128, in_channels = 3, out_channels = 16, depth, height, width = 16, 32, 32, kernel_size = 3, stride = 2` |
| 61 | `ConvTranspose3d` | `ConvTranspose3d_ReLU_GroupNorm` | `batch_size = 16, in_channels = 64, out_channels = 128, kernel_size = 3` |
| 62 | `GEMM/Linear` | `Matmul_GroupNorm_LeakyReLU_Sum` | `batch_size = 1024, input_size = 8192, hidden_size = 8192, num_groups = 512` |
| 63 | `GEMM/Linear` | `Gemm_ReLU_Divide` | `batch_size = 1024, in_features = 8192, out_features = 8192` |
| 64 | `GEMM/Linear` | `Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU` | `batch_size = 1024, in_features = 8192, out_features = 8192` |
| 65 | `Conv2d` | `Conv2d_AvgPool_Sigmoid_Sum` | `batch_size = 128, in_channels = 8, out_channels = 64, height, width = 384, 384, kernel_size = 3` |
| 66 | `GEMM/Linear` | `Matmul_Dropout_Softmax` | `batch_size = 128, in_features = 16384, out_features = 16384, dropout_p = 0.2` |
| 67 | `Conv2d` | `Conv2d_GELU_GlobalAvgPool` | `batch_size = 128, in_channels = 8, out_channels = 64, height, width = 256, 256, kernel_size = 3` |
| 68 | `GEMM/Linear` | `Matmul_Min_Subtract` | `batch_size = 128, in_features = 16384, out_features = 16384` |
| 69 | `Conv2d` | `Conv2d_HardSwish_ReLU` | `batch_size = 128, in_channels = 8, out_channels = 64, height, width = 128, 128, kernel_size = 3` |
| 70 | `GEMM/Linear` | `Gemm_Sigmoid_Scaling_ResidualAdd` | `batch_size = 1024, input_size = 8192, hidden_size = 8192` |
| 71 | `Conv2d` | `Conv2d_Divide_LeakyReLU` | `batch_size = 128, in_channels = 8, out_channels = 64, height, width = 128, 128, kernel_size = 3` |
| 72 | `ConvTranspose3d` | `ConvTranspose3d_BatchNorm_AvgPool_AvgPool` | `batch_size = 64, in_channels = 3, out_channels = 16, depth, height, width = 32, 32, 32, kernel_size = 3, stride = 2` |
| 73 | `Conv2d` | `Conv2d_BatchNorm_Scaling` | `batch_size = 128, in_channels = 8, out_channels = 64, height, width = 128, 128, kernel_size = 3` |
| 74 | `ConvTranspose3d` | `ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max` | `batch_size = 16, in_channels = 16, out_channels = 32, depth, height, width = 16, 32, 32, kernel_size = 3, stride = 2` |
| 75 | `GEMM/Linear` | `Gemm_GroupNorm_Min_BiasAdd` | `batch_size = 1024, in_features = 8192, out_features = 8192, num_groups = 512` |
| 76 | `GEMM/Linear` | `Gemm_Add_ReLU` | `batch_size = 1024, in_features = 8192, out_features = 8192` |
| 77 | `ConvTranspose3d` | `ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool` | `batch_size = 16, in_channels = 64, out_channels = 128, depth, height, width = 16, 32, 32, kernel_size = 5` |
| 78 | `ConvTranspose3d` | `ConvTranspose3d_Max_Max_Sum` | `batch_size = 16, in_channels = 32, out_channels = 64, depth, height, width = 32, 32, 32, kernel_size = 5, stride = 2` |
| 79 | `Conv3d` | `Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max` | `batch_size = 128, in_channels = 3, out_channels = 16, depth, height, width = 16, 32, 32, kernel_size = 3` |
| 80 | `GEMM/Linear` | `Gemm_Max_Subtract_GELU` | `batch_size = 1024, in_features = 8192, out_features = 8192` |
| 81 | `GEMM/Linear` | `Gemm_Swish_Divide_Clamp_Tanh_Clamp` | `batch_size = 1024, in_features = 8192, out_features = 8192` |
| 82 | `Conv2d` | `Conv2d_Tanh_Scaling_BiasAdd_Max` | `batch_size = 128, in_channels = 8, out_channels = 64, height, width = 256, 256, kernel_size = 3` |
| 83 | `Conv3d` | `Conv3d_GroupNorm_Min_Clamp_Dropout` | `batch_size = 128, in_channels = 3, out_channels = 16, depth, height, width = 16, 64, 64, kernel_size = 3, dropout_p = 0.2` |
| 84 | `GEMM/Linear` | `Gemm_BatchNorm_Scaling_Softmax` | `batch_size = 1024, in_features = 8192, out_features = 8192` |
| 85 | `Conv2d` | `Conv2d_GroupNorm_Scale_MaxPool_Clamp` | `batch_size = 128, in_channels = 8, out_channels = 64, height, width = 128, 128, kernel_size = 3, num_groups = 16` |
| 86 | `GEMM/Linear` | `Matmul_Divide_GELU` | `batch_size = 1024, input_size = 8192` |
| 87 | `Conv2d` | `Conv2d_Subtract_Subtract_Mish` | `batch_size = 128, in_channels = 8, out_channels = 64, height, width = 256, 256, kernel_size = 3` |
| 88 | `GEMM/Linear` | `Gemm_GroupNorm_Swish_Multiply_Swish` | `batch_size = 1024, in_features = 8192, out_features = 8192, num_groups = 256` |
| 89 | `ConvTranspose3d` | `ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max` | `batch_size = 128, in_channels = 3, out_channels = 16, depth, height, width = 16, 32, 32, kernel_size = 3, stride = 2` |
| 90 | `Conv3d` | `Conv3d_LeakyReLU_Sum_Clamp_GELU` | `batch_size = 128, in_channels = 8, out_channels = 64, depth, height, width = 16, 64, 64, kernel_size = 3` |
| 91 | `ConvTranspose2d` | `ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid` | `batch_size = 128, in_channels = 64, out_channels = 128, height, width = 64, 64, kernel_size = 4, stride = 2` |
| 92 | `Conv2d` | `Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp` | `batch_size = 128, in_channels = 8, out_channels = 64, height, width = 128, 128, kernel_size = 3` |
| 93 | `ConvTranspose2d` | `ConvTranspose2d_Add_Min_GELU_Multiply` | `batch_size = 128, in_channels = 64, out_channels = 128, height, width = 64, 64, kernel_size = 4, stride = 2` |
| 94 | `GEMM/Linear` | `Gemm_BiasAdd_Hardtanh_Mish_GroupNorm` | `batch_size = 1024, in_features = 8192, out_features = 8192, num_groups = 256` |
| 95 | `GEMM/Linear` | `Matmul_Add_Swish_Tanh_GELU_Hardtanh` | `batch_size = 1024, in_features = 8192, out_features = 8192` |
| 96 | `ConvTranspose3d` | `ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp` | `batch_size = 128, in_channels = 3, out_channels = 16, depth, height, width = 16, 32, 32, kernel_size = 3, stride = 2` |
| 97 | `GEMM/Linear` | `Matmul_BatchNorm_BiasAdd_Divide_Swish` | `batch_size = 1024, in_features = 8192, out_features = 8192` |
| 98 | `GEMM/Linear` | `Matmul_AvgPool_GELU_Scale_Max` | `batch_size = 1024, in_features = 8192, out_features = 8192` |
| 99 | `GEMM/Linear` | `Matmul_GELU_Softmax` | `batch_size = 1024, in_features = 8192, out_features = 8192` |
| 100 | `ConvTranspose3d` | `ConvTranspose3d_Clamp_Min_Divide` | `batch_size = 16, in_channels = 64, out_channels = 128, depth, height, width = 24, 48, 48, kernel_size = 3, stride = 2` |
