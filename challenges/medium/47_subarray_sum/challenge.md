Implement a program that computes the sum of a subarray of 32-bit integers. You are given an input array `input` of length `N`, and two indices `S` and `E`. `S` and `E` are inclusive, 0-based start and end indices — compute the sum of `input[S..E]`.

计算闭区间 `[S, E]` 内所有元素的和；索引从零开始，输入为长度 `N` 的 32 位整数数组。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` variable

## Example 1 / 示例 1:

    Input: input = [1, 2, 1, 3, 4], S = 1, E = 3
    Output: output = 6

## Example 2 / 示例 2:

    Input: input = [1, 2, 3, 4], S = 0, E = 3
    Output: output = 10

## Constraints / 约束

- 1 ≤ `N` ≤ 100,000,000
- 1 ≤ `input[i]` ≤ 10
- 0 ≤ `S` ≤ `E` ≤ `N - 1`
- Performance is measured with `N` = 100,000,000
