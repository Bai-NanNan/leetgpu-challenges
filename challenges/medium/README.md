# Medium Challenges

本目录包含 57 道 GPU 算子题，按主要实现模式大致分为以下几类：

| 类型 | 题目 |
| --- | --- |
| 归约、扫描与统计 | Reduction、Histogramming、Prefix Sum、Dot Product、Monte Carlo Integration、Count Element（1D/2D/3D）、Subarray Sum（1D/2D/3D）、Max Subarray Sum、FP16 Dot Product、Segmented Prefix Sum |
| Softmax、损失与归一化 | Softmax、Categorical Cross Entropy、MSE、Batch Normalization、RMS Normalization、Group Normalization |
| 矩阵乘法与线性代数 | Sparse Matrix-Vector Multiplication、GEMM、Batched MatMul、INT8 MatMul、Matrix Power、FP16 Batched MatMul、Sparse-Dense MatMul、INT4 MatMul |
| 经典机器学习算法 | Ordinary Least Squares、Logistic Regression |
| 卷积、池化、Stencil 与频域变换 | 2D/3D Convolution、Gaussian Blur、2D Max Pooling、Jacobi Stencil、2D FFT、Causal Depthwise Conv1D |
| 选择、排序、搜索与压缩 | Top-K、Nearest Neighbor、Top-P Sampling、MoE Top-K Gating、Parallel Merge、Stream Compaction |
| Attention、位置编码与序列递推 | Softmax Attention、ALiBi Attention、RoPE、Grouped Query Attention、Linear Recurrence、Speculative Decoding、Decaying Causal Attention、SSM Selective Scan、INT8 KV-Cache Attention |
| Transformer/模型组件与量化辅助 | Weight Dequantization、Adder Transformer、SwiGLU MLP、LoRA Linear、Token Embedding Layer |

题目目录按 `<编号>_<名称>` 命名；每题通常包含题目说明、参考实现和对应的 CUDA/Triton starter 模板。
