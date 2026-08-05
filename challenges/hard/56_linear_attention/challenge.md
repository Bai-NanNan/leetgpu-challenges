Implement **Linear Attention** for a given set of matrices, following the method described in <a href="https://arxiv.org/pdf/2006.16236" target="_blank">"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"</a> . Given the query matrix `Q` of size `M×d`, key matrix `K` of size `M×d`, and value matrix `V` of size `M×d`, your program should compute the output matrix using the formula: 

按照 <a href="https://arxiv.org/pdf/2006.16236" target="_blank">《Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention》</a> 中介绍的方法，实现给定矩阵的**线性注意力**。给定大小为 `M×d` 的查询矩阵 `Q`、键矩阵 `K` 和值矩阵 `V`，程序应使用下式计算输出矩阵：

$$
\text{LinearAttention}(Q, K, V) = \frac{\phi(Q) \left(\phi(K)^T V \right)}{\phi(Q) \left(\sum_j \phi(K_j) \right)}
$$

where $\phi(x)$ is a feature map applied element-wise, for example: 

其中 $\phi(x)$ 是逐元素应用的特征映射，例如：

$$
\phi(x) = \text{ELU}(x) + 1 =
  \begin{cases}
  x + 1, & x > 0 \\
  e^x, & x \le 0
  \end{cases}
$$

 All matrices `Q`, `K`, `V`, and `output` are of type `float32`, and `M` and `d` are of type `int32`.

矩阵 `Q`、`K`、`V` 和 `output` 的类型均为 `float32`，`M` 和 `d` 的类型为 `int32`。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted) / 只能使用原生功能（不得使用外部库）
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in the output matrix `output` / 最终结果必须存储在输出矩阵 `output` 中

## Example 1 / 示例 1

**Input:**\
`Q` (2×4): 

$$
\begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 0.0
\end{bmatrix}
$$

 `K` (2×4): 

$$
\begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 0.0
\end{bmatrix}
$$

 `V` (2×4): 

$$
\begin{bmatrix}
1.0 & 2.0 & 3.0 & 4.0 \\
5.0 & 6.0 & 7.0 & 8.0
\end{bmatrix}
$$

**Output:**\
`output` (2×4): 

$$
\begin{bmatrix}
2.8461537 & 3.8461537 & 4.8461537 & 5.8461537 \\
3.1538463 & 4.1538463 & 5.1538463 & 6.1538463
\end{bmatrix}
$$

## Example 2 / 示例 2

**Input:**\
`Q` (2×2): 

$$
\begin{bmatrix}
0.0 & 0.0 \\
1.0 & 1.0
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
`output` (2×2): 

$$
\begin{bmatrix}
4.0 & 5.0 \\
4.0 & 5.0
\end{bmatrix}
$$

## Constraints / 约束

- Matrix `Q`, `K`, and `V` are all of size `M×d` / 矩阵 `Q`、`K` 和 `V` 的大小均为 `M×d`
- 1 ≤ `M` ≤ 10000 / `M` 的范围为 1 至 10000
- 1 ≤ `d` ≤ 128 / `d` 的范围为 1 至 128
- All elements in `Q`, `K`, and `V` are sampled from`[-100.0, 100.0]` / `Q`、`K` 和 `V` 中的元素均从 `[-100.0, 100.0]` 范围采样
- Data type for all matrices is `float32` / 所有矩阵的数据类型均为 `float32`
- Performance is measured with `M` = 10,000 / 性能测试使用 `M` = 10,000
