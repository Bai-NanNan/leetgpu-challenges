Implement a GPU program that computes the Fast Fourier Transform (FFT) of a complex-valued 1-D signal. Given an input `signal` array containing `N` complex numbers stored as interleaved real/imaginary pairs, compute the discrete Fourier transform and store the result in the `spectrum` array. The FFT converts a time-domain signal into its frequency-domain representation using the formula: 

实现一个 GPU 程序，计算复数值一维信号的快速傅里叶变换（FFT）。给定输入数组 `signal`，其中包含以交错实部/虚部形式存储的 `N` 个复数，计算离散傅里叶变换，并将结果存储在 `spectrum` 数组中。FFT 使用下式将时域信号转换为频域表示：

$$
X_k = \sum_{n=0}^{N-1}
  x_n \cdot e^{-j 2\pi kn / N} \quad \text{for } k = 0, 1, \ldots, N-1
$$

 The FFT algorithm reduces the computational complexity from O(N²) to O(N log N) by exploiting symmetries in the twiddle factors.

FFT 算法利用旋转因子的对称性，将计算复杂度从 O(N²) 降低到 O(N log N)。

## Implementation Requirements / 实现要求

- External libraries (cuFFT etc.) are not permitted / 不得使用外部库（如 cuFFT）
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in the `spectrum` array / 最终结果必须存储在 `spectrum` 数组中
- The kernel must be entirely GPU-resident—no host-side FFT calls / 内核必须完全驻留在 GPU 上，不得调用主机端 FFT
- Both input and output use interleaved real/imaginary layout: `[real₀, imag₀, real₁, imag₁, ...]` / 输入和输出均使用交错实部/虚部布局：`[real₀, imag₀, real₁, imag₁, ...]`

## Example 1 / 示例 1

    Input:  N = 4
            signal = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            (represents: [1+0j, 0+0j, 0+0j, 0+0j])

    Output: spectrum = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
            (represents: [1+0j, 1+0j, 1+0j, 1+0j])

## Example 2 / 示例 2

    Input:  N = 2
            signal = [1.0, 0.0, 1.0, 0.0]
            (represents: [1+0j, 1+0j])

    Output: spectrum = [2.0, 0.0, 0.0, 0.0]
            (represents: [2+0j, 0+0j])

## Constraints / 约束

- `1 ≤ N ≤ 262,144` / `N` 的范围为 1 至 262,144
- All values are 32-bit floating point numbers / 所有值均为 32 位浮点数
- Absolute error ≤ 1e-3 and relative error ≤ 1e-3 / 绝对误差不超过 1e-3，且相对误差不超过 1e-3
- Input and output arrays have length `2 × N` / 输入和输出数组长度均为 `2 × N`
- Performance is measured with `N` = 262,144 / 性能测试使用 `N` = 262,144
