Implement a program for multi-head self-attention. Given three input matrices $Q$ (queries), $K$ (keys), and $V$ (values) of size $N \times d_{\text{model}}$, compute: 

编写一个程序，实现多头自注意力。给定三个大小为 $N \times d_{\text{model}}$ 的输入矩阵 $Q$（查询）、$K$（键）和 $V$（值），计算：

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)
$$

 where each head computes: 

其中每个注意力头的计算方式为：

$$
\text{head}_i = \text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i
$$

 with $d_k = d_{\text{model}}/h$ and $Q_i, K_i, V_i$ being the i-th head's partition of the input matrices.

其中 $d_k = d_{\text{model}}/h$，$Q_i$、$K_i$ 和 $V_i$ 分别是输入矩阵按第 $i$ 个注意力头划分后的部分。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted) / 只能使用原生功能（不得使用外部库）
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in the `output` array / 最终结果必须存储在 `output` 数组中

## Example 1 / 示例 1

Input: 

$$
\begin{align*}
N &= 2, \quad d_{\text{model}} = 4, \quad h = 2 \\[1em]
Q &= \begin{bmatrix}
1.0 & 0.0 & 2.0 & 3.0 \\
4.0 & 5.0 & 6.0 & 7.0
\end{bmatrix} \\[1em]
K &= \begin{bmatrix}
1.0 & 2.0 & 3.0 & 4.0 \\
5.0 & 6.0 & 7.0 & 8.0
\end{bmatrix} \\[1em]
V &= \begin{bmatrix}
0.5 & 1.0 & 1.5 & 2.0 \\
2.5 & 3.0 & 3.5 & 4.0
\end{bmatrix}
\end{align*}
$$

 Output: 

$$
\begin{bmatrix}
2.39 & 2.89 & 3.50 & 4.00 \\
2.50 & 3.00 & 3.50 & 4.00
\end{bmatrix}
$$

## Example 2 / 示例 2

Input: 

$$
\begin{align*}
N &= 1, \quad d_{\text{model}} = 2, \quad h = 1 \\[1em]
Q &= \begin{bmatrix} 1.0 & 1.0 \end{bmatrix} \\[1em]
K &= \begin{bmatrix} 1.0 & 1.0 \end{bmatrix} \\[1em]
V &= \begin{bmatrix} 2.0 & 3.0 \end{bmatrix}
\end{align*}
$$

 Output: 

$$
\begin{bmatrix} 2.0 & 3.0 \end{bmatrix}
$$

## Constraints / 约束

- `1 ≤ N ≤ 10000` / `N` 的范围为 1 至 10000
- `2 ≤ d_model ≤ 1024` / `d_model` 的范围为 2 至 1024
- `1 ≤ h ≤ d_model` / `h` 的范围为 1 至 `d_model`
- `d_model % h == 0` / `d_model` 必须能被 `h` 整除
- `-10.0 ≤ values ≤ 10.0` / `values` 的范围为 -10.0 至 10.0
- Performance is measured with `N` = 1,024, `d_model` = 1,024 / 性能测试使用 `N` = 1,024、`d_model` = 1,024
