Implement a GPU program that, for `N` three-dimensional points stored on the device, fills `indices[i]` with the index `j ≠ i` of the point closest to `points[i]`. Comparing *squared* Euclidean distance is sufficient—you do **not** need to compute square-roots.

对于每个三维点 `points[i]`，找出与它距离最近且索引不等于 `i` 的点 `j`，并将 `j` 写入 `indices[i]`。比较平方欧氏距离即可，无需开平方。

## Implementation Requirements / 实现要求

- The `solve` function signature must remain unchanged
- External libraries are not permitted
- The final result must be stored in the `indices` array

## Example 1 / 示例 1:

    Input:  points  = [(0,0,0), (1,0,0), (5,5,5)]
            indices = [-1, -1, -1]
            N       = 3
    Output: indices = [1, 0, 1]   # 0⇆1 are nearest, 2 is closest to 1

## Constraints / 约束

- 1 ≤ `N` ≤ 100,000
- Coordinates are 32-bit floats in the range \[-1000, 1000\]
- Performance is measured with `N` = 10,000
