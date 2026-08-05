Write a GPU program that applies the sigmoid activation function element-wise to a vector of 32-bit floating point numbers. For each element `x` in the input vector `X`, compute `sigmoid(x) = 1 / (1 + exp(-x))` and store the result in the output vector `Y`. The sigmoid function maps any real number to the range (0, 1).

编写一个 GPU 程序，对由 32 位浮点数组成的向量逐元素应用 sigmoid 激活函数。对于输入向量 `X` 中的每个元素 `x`，计算 `sigmoid(x) = 1 / (1 + exp(-x))`，并将结果存储在输出向量 `Y` 中。sigmoid 函数将任意实数映射到 (0, 1) 区间。

## Implementation Requirements / 实现要求

- External libraries are not permitted / 不得使用外部库
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in vector `Y` / 最终结果必须存储在向量 `Y` 中

## Example 1 / 示例 1

    Input:  X = [0.0, 1.0, -1.0, 2.0]
    Output: Y = [0.5, 0.7311, 0.2689, 0.8808]

## Example 2 / 示例 2

    Input:  X = [0.5, -0.5, 3.0, -3.0]
    Output: Y = [0.6225, 0.3775, 0.9526, 0.0474]

## Constraints / 约束

- 1 ≤ `N` ≤ 100,000,000 / `N` 的范围为 1 至 100,000,000
- Input values are finite 32-bit floating point numbers / 输入值为有限的 32 位浮点数
- Performance is measured with `N` = 50,000,000 / 性能测试使用 `N` = 50,000,000
