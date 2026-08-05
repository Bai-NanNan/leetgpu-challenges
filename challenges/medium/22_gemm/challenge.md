Implement a basic General Matrix Multiplication (GEMM). Given matrix $A$ of dimensions $M \times K$, matrix $B$ of dimensions $K \times N$, input/output matrix $C$ of dimensions $M \times N$, and scalar multipliers $\alpha$ and $\beta$, compute the operation: 

$$
C = \alpha \cdot (A \times B) + \beta \cdot C_{initial}
$$

The input matrices $A$, $B$, and the initial state of $C$ contain 16-bit floating-point numbers (FP16/`half`). All matrices are stored in row-major order. The scalars $\alpha$ and $\beta$ are 32-bit floats.

实现带缩放项的 GEMM：先计算 `A × B`，再与初始矩阵 `C` 按 `alpha` 和 `beta` 组合。乘法累加使用 FP32，最终结果转换并写回 FP16 的 `C`。

## Implementation Requirements / 实现要求

- Use only native features (external libraries other than WMMA are not permitted).
- The `solve` function signature must remain unchanged.
- Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16.
- The final result must be stored back into matrix `C` as `half`.

## Example / 示例:

Input:\
*(Note: Input matrices A, B, C_initial are FP16 type for the problem)*\
Matrix $A$ ($M=2, K=3$): 

$$
\begin{bmatrix}
1.0 & 2.0 & 3.0 \\
4.0 & 5.0 & 6.0
\end{bmatrix}
$$

 Matrix $B$ ($K=3, N=2$): 

$$
\begin{bmatrix}
1.0 & 2.0 \\
3.0 & 4.0 \\
5.0 & 6.0
\end{bmatrix}
$$

 Matrix $C_{initial}$ ($M=2, N=2$): 

$$
\begin{bmatrix}
1.0 & 1.0 \\
1.0 & 1.0
\end{bmatrix}
$$

 

$$
\alpha = 1.0 \text{ (FP32)}
$$

 

$$
\beta = 0.0 \text{ (FP32)}
$$

 Output (FP16):\
Matrix $C$ ($M=2, N=2$): 

$$
\begin{bmatrix}
22.0 & 28.0 \\
49.0 & 64.0
\end{bmatrix}
$$

## Constraints / 约束

- 16 ≤ `M`, `N`, `K` ≤ 4096
- Performance is measured with `K` = 1,024, `M` = 1,024, `N` = 1,024
