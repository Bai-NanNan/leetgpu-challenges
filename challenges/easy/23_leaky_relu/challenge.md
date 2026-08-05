Implement a program that performs the leaky ReLU activation function on a vector of floating-point numbers. The leaky ReLU function is defined as: 

编写一个程序，对由浮点数组成的向量执行 Leaky ReLU 激活函数。Leaky ReLU 函数定义如下：

$$
f(x) = \begin{cases}
      x & \text{if } x > 0 \\
      \alpha x & \text{if } x \leq 0
    \end{cases}
$$

 where $\alpha$ is a small positive constant (0.01 in this problem).

其中 $\alpha$ 是一个较小的正数（本题中为 0.01）。

## Implementation Requirements / 实现要求

- External libraries are not permitted / 不得使用外部库
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in vector `output` / 最终结果必须存储在向量 `output` 中
- Use $\alpha = 0.01$ as the leaky coefficient / 使用 $\alpha = 0.01$ 作为泄漏系数

## Example 1 / 示例 1

      Input:  x = [1.0, -2.0, 3.0, -4.0]
      Output: y = [1.0, -0.02, 3.0, -0.04]

## Example 2 / 示例 2

      Input:  x = [-1.5, 0.0, 2.5, -3.0]
      Output: y = [-0.015, 0.0, 2.5, -0.03]

## Constraints / 约束

- 1 ≤ `N` ≤ 100,000,000 / `N` 的范围为 1 至 100,000,000
- -1000.0 ≤ `input[i]` ≤ 1000.0 / `input[i]` 的范围为 -1000.0 至 1000.0
- Performance is measured with `N` = 50,000,000 / 性能测试使用 `N` = 50,000,000
