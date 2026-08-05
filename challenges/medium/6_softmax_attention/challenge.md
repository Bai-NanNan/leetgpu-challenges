Implement a GPU program that computes the softmax attention operation for a given set of matrices. Given the query matrix `Q` of size `M×d`, key matrix `K` of size `N×d`, and value matrix `V` of size `N×d`, your program should compute the output matrix using the formula: 

$$
\text{Attention}(Q, K, V) = \text{softmax}\Bigl( \frac{QK^T}{\sqrt{d}} \Bigr)V,
$$

 where the softmax function is applied row-wise.

按公式计算缩放点积注意力：先求 `QK^T / sqrt(d)`，对每一行执行 softmax，再与值矩阵 `V` 相乘。

## Implementation Requirements / 实现要求

- Use only GPU native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the output matrix `output`

## Example 1 / 示例 1:

**Input:**\
`Q` (2×4): 

$$
\begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 0.0
\end{bmatrix}
$$

 `K` (3×4): 

$$
\begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 0.0 \\
0.0 & 0.0 & 1.0 & 0.0
\end{bmatrix}
$$

 `V` (3×4): 

$$
\begin{bmatrix}
1.0 & 2.0 & 3.0 & 4.0 \\
5.0 & 6.0 & 7.0 & 8.0 \\
9.0 & 10.0 & 11.0 & 12.0
\end{bmatrix}
$$

**Output:**\
`output` (2×4): 

$$
\begin{bmatrix}
4.29 & 5.29 & 6.29 & 7.29 \\
5.00 & 6.00 & 7.00 & 8.00
\end{bmatrix}
$$

## Example 2 / 示例 2:

**Input:**\
`Q` (1×2): 

$$
\begin{bmatrix}
1.0 & 2.0
\end{bmatrix}
$$

 `K` (2×2): 

$$
\begin{bmatrix}
1.0 & 0.0 \\
0.0 & 1.0
\end{bmatrix}
$$

 `V` (2×2): 

$$
\begin{bmatrix}
3.0 & 4.0 \\
5.0 & 6.0
\end{bmatrix}
$$

**Output:**\
`output` (1×2): 

$$
\begin{bmatrix}
4.34 & 5.34
\end{bmatrix}
$$

## Constraints / 约束

- Matrix `Q` is of size `M×d` and matrices `K` and `V` are of size `N×d`
- 1 ≤ `M`, `N` ≤ 100,000
- 1 ≤ `d` ≤ 128
- Performance is measured with `M` = 512, `N` = 256
