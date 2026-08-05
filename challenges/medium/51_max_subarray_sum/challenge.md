Implement a program that computes the maximum sum of any contiguous subarray of length exactly `window_size`. You are given an array `input` of length `N` consisting of 32-bit signed integers, and an integer `window_size`.

在整数数组中寻找长度恰好为 `window_size` 的连续子数组，并输出其中最大的元素和。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` variable

## Example 1 / 示例 1:

    Input:  input = [1, 2, 4, 2, 3], window_size = 2
    Output: output = 6

## Example 2 / 示例 2:

    Input:  input = [-1, -4, -2, 1], window_size = 3
    Output: output = -5

## Constraints / 约束

- 1 ≤ `N` ≤ 50,000
- -10 ≤ `input[i]` ≤ 10
- 1 ≤ `window_size` ≤ `N`
- Performance is measured with `N` = 50,000
