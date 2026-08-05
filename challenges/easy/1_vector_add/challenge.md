Write a GPU program that performs element-wise addition of two vectors containing 32-bit floating point numbers. The program should take two input vectors of equal length and produce a single output vector containing their sum.

编写一个 GPU 程序，对两个包含 32 位浮点数的向量执行逐元素加法。程序应接收两个长度相同的输入向量，并生成一个包含它们之和的输出向量。

## Implementation Requirements / 实现要求

- External libraries are not permitted / 不得使用外部库
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in vector `C` / 最终结果必须存储在向量 `C` 中

## Example 1 / 示例 1

    Input:  A = [1.0, 2.0, 3.0, 4.0]
            B = [5.0, 6.0, 7.0, 8.0]
    Output: C = [6.0, 8.0, 10.0, 12.0]

## Example 2 / 示例 2

    Input:  A = [1.5, 1.5, 1.5]
            B = [2.3, 2.3, 2.3]
    Output: C = [3.8, 3.8, 3.8]

## Constraints / 约束

- Input vectors `A` and `B` have identical lengths / 输入向量 `A` 和 `B` 的长度相同
- 1 ≤ `N` ≤ 100,000,000 / `N` 的范围为 1 至 100,000,000
- Performance is measured with `N` = 25,000,000 / 性能测试使用 `N` = 25,000,000
