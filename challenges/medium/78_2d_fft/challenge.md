Compute the 2D Discrete Fourier Transform (2D DFT) of a complex-valued signal stored on the GPU. Given a 2D complex input signal of shape `M × N`, compute its 2D DFT spectrum using the row-column decomposition: apply a 1D DFT along each row, then a 1D DFT along each column of the result. All values are 32-bit floating point.

对 GPU 上的复数二维信号执行二维离散傅里叶变换：先沿每一行做一维 DFT，再沿中间结果的每一列做一维 DFT，得到频谱。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in `spectrum`
- The input and output are stored as 1D arrays of interleaved real and imaginary parts in row-major order: element `x[m, n]` has its real part at index `2*(m*N + n)` and imaginary part at index `2*(m*N + n) + 1`

## Example / 示例

Input: `M` = 2, `N` = 2\
Signal $x[m, n]$ (real part): 

$$
\begin{bmatrix}
1.0 & 0.0 \\
0.0 & 0.0
\end{bmatrix}
$$

 Signal $x[m, n]$ (imaginary part): 

$$
\begin{bmatrix}
0.0 & 0.0 \\
0.0 & 0.0
\end{bmatrix}
$$

 Output:\
Spectrum $X[k, l]$ (real part): 

$$
\begin{bmatrix}
1.0 & 1.0 \\
1.0 & 1.0
\end{bmatrix}
$$

 Spectrum $X[k, l]$ (imaginary part): 

$$
\begin{bmatrix}
0.0 & 0.0 \\
0.0 & 0.0
\end{bmatrix}
$$

## Constraints / 约束

- 1 ≤ `M`, `N` ≤ 4096
- Signal values are 32-bit floating point (real and imaginary parts)
- Performance is measured with `M` = 2,048, `N` = 2,048
