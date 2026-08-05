Write a GPU program that counts the number of elements with the integer value p in an 3D array of 32-bit integers. The program should count the number of elements with p in an 3D array. You are given an input 3D array `input` of length `N x M x K` and integer `p`.

遍历 `N × M × K` 的三维整数数组，统计值等于 `p` 的元素数量，并将计数结果写入输出变量。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` variable

## Example 1 / 示例 1:

    Input: input [[[1, 2, 3],
                   [4, 5, 1]],
                  [[1, 1, 1],
                   [2, 2, 2]]]
           N = 2, M = 2, K = 3
           p = 1
    Output: output = 5

## Example 2 / 示例 2:

    Input: input [[[5, 10],
                   [5, 2],
                   [2, 2]]]
           N = 1, M = 3, K = 2
           p = 1
    Output: output = 0

## Constraints / 约束

- 1 ≤ `N, M, K` ≤ 1,000
- 1 ≤ `input[i], p` ≤ 100
- Performance is measured with `K` = 500, `M` = 500, `N` = 500
