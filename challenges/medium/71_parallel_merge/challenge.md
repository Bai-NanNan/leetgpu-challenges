Given two sorted arrays `A` of length `M` and `B` of length `N`, both containing 32-bit floating-point values in non-decreasing order, produce a single sorted array `C` of length `M + N` containing all elements of `A` and `B` in non-decreasing order.

给定两个已按非递减顺序排列的浮点数组 `A` 和 `B`，合并它们的全部元素，生成同样按非递减顺序排列的 `C`。

## Implementation Requirements / 实现要求

- Use only GPU native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final merged result must be stored in `C`

## Example / 示例

    Input:
      A = [1.0, 3.0, 5.0, 7.0],  M = 4
      B = [2.0, 4.0, 6.0, 8.0],  N = 4

    Output:
      C = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    Input:
      A = [-1.0, 1.0, 3.0],  M = 3
      B = [2.0],             N = 1

    Output:
      C = [-1.0, 1.0, 2.0, 3.0]

## Constraints / 约束

- 1 ≤ `M`, `N` ≤ 50,000,000
- `M + N` ≤ 50,000,000
- Both `A` and `B` are sorted in non-decreasing order
- Elements are 32-bit floats
- Performance is measured with `M` = 25,000,000, `N` = 25,000,000
