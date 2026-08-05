Implement a GPU program that performs sparse matrix-vector multiplication. Given a sparse matrix $A$ of dimensions $M \times N$ and a dense vector $x$ of length $N$, compute the product vector $y = A \times x$, which will have length $M$. `A` is stored in row-major order. `nnz` is the number of non-zero elements in `A`.

Mathematically, the operation is defined as: 

$$
y_i = \sum_{j=0}^{N-1} A_{ij} \cdot x_j \quad \text{for} \quad i = 0, 1, \ldots, M-1
$$

The matrix $A$ is approximately 60 - 70% sparse.

矩阵 `A` 是约 60%–70% 元素为零的稀疏矩阵，向量 `x` 是稠密向量。对每一行计算非零矩阵元素与 `x` 的乘积之和，得到长度为 `M` 的向量 `y`。

## Implementation Requirements / 实现要求

- Use only GPU native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in vector `y`

## Example / 示例:

Input:\
Matrix $A$ ($3 \times 4$): 

$$
\begin{bmatrix}
5.0 & 0.0 & 0.0 & 1.0 \\
0.0 & 2.0 & 3.0 & 0.0 \\
0.0 & 0.0 & 0.0 & 4.0
\end{bmatrix}
$$

 Vector $x$: 

$$
\begin{bmatrix}
1.0 \\
2.0 \\
3.0 \\
4.0
\end{bmatrix}
$$

 Output:\
Vector $y$: 

$$
\begin{bmatrix}
9.0 \\
13.0 \\
16.0
\end{bmatrix}
$$

## Constraints / 约束

- 1 ≤ `M`, `N` ≤ 10,000
- The matrix $A$ is approximately 60-70% sparse (i.e., 60-70% of elements are zero)
- Performance is measured with `M` = 1,000, `N` = 10,000
