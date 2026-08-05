Write a GPU program that computes the histogram of an array of 32-bit integers. The histogram should count the number of occurrences of each integer value in the range `[0, num_bins)`. You are given an input array `input` of length `N` and the number of bins `num_bins`.

The result should be an array of integers of length `num_bins`, where each element represents the count of occurrences of its corresponding index in the input array.

给定长度为 `N` 的整数数组和 bin 数量 `num_bins`，统计每个 `[0, num_bins)` 范围内整数出现的次数，并将长度为 `num_bins` 的计数结果写入 `histogram`。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `histogram` array.

## Examples / 示例

    Input: input = [0, 1, 2, 1, 0],  N = 5, num_bins = 3
    Output: [2, 2, 1]

    Input: input = [3, 3, 3, 3], N = 4, num_bins = 5
    Output: [0, 0, 0, 4, 0]

## Constraints / 约束

- 1 ≤ `N` ≤ 100,000,000
- 0 ≤ `input[i]` \< `num_bins`
- 1 ≤ `num_bins` ≤ 1024
- Performance is measured with `N` = 50,000,000, `num_bins` = 256
