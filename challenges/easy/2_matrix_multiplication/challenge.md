Write a program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix $A$ of dimensions $M \times N$ and matrix $B$ of dimensions $N \times K$, compute the product matrix $C = A \times B$, which will have dimensions $M \times K$. All matrices are stored in row-major format.

编写一个 GPU 程序，计算两个 32 位浮点矩阵的乘积。给定尺寸为 $M \times N$ 的矩阵 $A$ 和尺寸为 $N \times K$ 的矩阵 $B$，计算乘积矩阵 $C = A \times B$，其尺寸为 $M \times K$。所有矩阵均以行优先格式存储。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted) / 只能使用原生功能（不得使用外部库）
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in matrix `C` / 最终结果必须存储在矩阵 `C` 中

## Example 1 / 示例 1

Input:\
Matrix $A$ ($2 \times 2$): 

$$
\begin{bmatrix}
1.0 & 2.0 \\
3.0 & 4.0
\end{bmatrix}
$$

 Matrix $B$ ($2 \times 2$): 

$$
\begin{bmatrix}
5.0 & 6.0 \\
7.0 & 8.0
\end{bmatrix}
$$

 Output:\
Matrix $C$ ($2 \times 2$): 

$$
\begin{bmatrix}
19.0 & 22.0 \\
43.0 & 50.0
\end{bmatrix}
$$

## Example 2 / 示例 2

Input:\
Matrix $A$ ($1 \times 3$): 

$$
\begin{bmatrix}
1.0 & 2.0 & 3.0
\end{bmatrix}
$$

 Matrix $B$ ($3 \times 1$): 

$$
\begin{bmatrix}
4.0 \\
5.0 \\
6.0
\end{bmatrix}
$$

 Output:\
Matrix $C$ ($1 \times 1$): 

$$
\begin{bmatrix}
32.0
\end{bmatrix}
$$

## Constraints / 约束

- 1 ≤ `M`, `N`, `K` ≤ 8192 / `M`、`N`、`K` 的范围均为 1 至 8192
- Performance is measured with `M` = 8192, `N` = 6144, `K` = 4096 / 性能测试使用 `M` = 8192、`N` = 6144、`K` = 4096
