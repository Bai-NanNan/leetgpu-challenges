Implement the Swish-Gated Linear Unit (SWiGLU) activation function forward pass for 1D input vectors. Given an input tensor of shape \[N\] where N is the number of elements, compute the output using the elementwise formula. The input and output tensor must be of type `float32`.

为一维输入向量实现 Swish 门控线性单元（SWiGLU）激活函数的前向计算。给定形状为 \[N\] 的输入张量（其中 N 为元素数量），使用逐元素公式计算输出。输入和输出张量必须为 `float32` 类型。

SWiGLU is defined as:

SWiGLU 定义如下：

1.  Split input $x$ into two halves: $x_1$ and $x_2$ / 将输入 $x$ 分成两半：$x_1$ 和 $x_2$
2.  Compute SiLU on the first half: / 对前半部分计算 SiLU：

$$
\text{SiLU}(x_1) = x_1 \cdot \sigma(x_1), \quad
        \sigma(x) = \frac{1}{1 + e^{-x}}
$$

3.  Compute the SWiGLU output: / 计算 SWiGLU 输出：

$$
\text{SWiGLU}(x_1, x_2) = \text{SiLU}(x_1) \cdot x_2
$$

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted) / 只能使用原生功能（不得使用外部库）
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in the `output` tensor / 最终结果必须存储在 `output` 张量中

## Example 1 / 示例 1

    Input:  [1.0, 2.0, 3.0, 4.0]  (N=4)
    Output: [2.1931758, 7.0463767]

## Example 2 / 示例 2

    Input:  [0.5, 1.0]  (N=2)
    Output: [0.31122968]

## Constraints / 约束

- 1 ≤ `N` ≤ 100,000 / `N` 的范围为 1 至 100,000
- N is an even number / N 必须为偶数
- -100.0 ≤ input values ≤ 100.0 / 输入值的范围为 -100.0 至 100.0
- Performance is measured with `N` = 100,000 / 性能测试使用 `N` = 100,000
