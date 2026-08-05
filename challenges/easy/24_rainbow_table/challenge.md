Implement a program that performs `R` rounds of parallel hashing on an array of 32-bit integers using the provided hash function. The hash should be applied `R` times iteratively (the output of one round becomes the input to the next).

使用给定的哈希函数，编写一个程序对 32 位整数数组执行 `R` 轮并行哈希。哈希函数应迭代应用 `R` 次（上一轮的输出作为下一轮的输入）。

## Implementation Requirements / 实现要求

- External libraries are not permitted / 不得使用外部库
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in array `output` / 最终结果必须存储在数组 `output` 中

## Example 1 / 示例 1

    Input:  numbers = [123, 456, 789], R = 2
    Output: hashes = [1636807824, 1273011621, 2193987222]

## Example 2 / 示例 2

    Input:  numbers = [0, 1, 2147483647], R = 3
    Output: hashes = [96754810, 3571711400, 2006156166]

## Constraints / 约束

- 1 ≤ `N` ≤ 10,000,000 / `N` 的范围为 1 至 10,000,000
- 1 ≤ `R` ≤ 100 / `R` 的范围为 1 至 100
- 0 ≤ `input[i]` ≤ 2147483647 / `input[i]` 的范围为 0 至 2147483647
- Performance is measured with `N` = 5,000,000 / 性能测试使用 `N` = 5,000,000
