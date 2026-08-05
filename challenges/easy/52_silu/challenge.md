Implement the SiLU (Sigmoid Linear Unit) activation function forward pass for 1D input vectors. Given an input tensor of shape \[N\] where N is the number of elements, compute the output using the elementwise formula.

为一维输入向量实现 SiLU（Sigmoid Linear Unit，Sigmoid 线性单元）激活函数的前向计算。给定形状为 \[N\] 的输入张量（其中 N 为元素数量），使用逐元素公式计算输出。

SiLU is defined as: 

SiLU 定义如下：

$$
\begin{align}
  \sigma(x) &= \frac{1}{1 + e^{-x}} \\
  \text{SiLU}(x) &= x \cdot \sigma(x)
  \end{align}
$$

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted) / 只能使用原生功能（不得使用外部库）
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in the `output` tensor / 最终结果必须存储在 `output` 张量中

## Example 1 / 示例 1

    Input:  input = [0.5, 1.0, -0.5]  (N=3)
    Output: output = [0.3112295, 0.731059, -0.1887705]

## Example 2 / 示例 2

    Input:  input = [-1.0, -2.0, -3.0, -4.0, -5.0]  (N=5)
    Output: output = [-0.26894143 -0.23840584 -0.14227763 -0.07194484 -0.03346425]

## Constraints / 约束

- 1 ≤ `N` ≤ 10,000 / `N` 的范围为 1 至 10,000
- -100.0 ≤ input values ≤ 100.0 / 输入值的范围为 -100.0 至 100.0
- Performance is measured with `N` = 50,000 / 性能测试使用 `N` = 50,000
