Write a GPU program that counts the number of elements with the integer value k in an array of 32-bit integers. The program should count the number of elements with k in an array. You are given an input array `input` of length `N` and integer `k`.

统计 32 位整数数组中等于给定整数 `k` 的元素数量，并将计数结果写入输出变量。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` variable

## Example 1 / 示例 1:

    Input: [1, 2, 3, 4, 1], k = 1
    Output: 2

## Example 2 / 示例 2:

    Input: [5, 10, 5, 2], k = 11
    Output: 0

## Constraints / 约束

- 1 ≤ `N` ≤ 100,000,000
- 1 ≤ `input[i], k` ≤ 100,000
- Performance is measured with `K` = 501,010, `N` = 100,000,000
