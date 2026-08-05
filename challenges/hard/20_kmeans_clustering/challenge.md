Implement the k-means clustering algorithm for 2D points. Given arrays of x and y coordinates for data points, initial centroids, and other parameters, assign each point to the nearest centroid and update the centroids iteratively. The final centroids and labels should be stored in the output arrays.

实现适用于二维点的 K-means 聚类算法。给定数据点的 x、y 坐标数组、初始质心和其他参数，将每个点分配给最近的质心，并迭代更新质心。最终质心和标签应存储在输出数组中。

## Implementation Requirements / 实现要求

- External libraries are not permitted / 不得使用外部库
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in `labels`, `final_centroid_x`, and `final_centroid_y` / 最终结果必须存储在 `labels`、`final_centroid_x` 和 `final_centroid_y` 中

## Example 1 / 示例 1

    Input:
    sample_size = 4, k = 2, max_iterations = 10
    data_x = [1.0, 2.0, 8.0, 9.0]
    data_y = [1.0, 2.0, 8.0, 9.0]
    initial_centroid_x = [1.0, 8.0]
    initial_centroid_y = [1.0, 8.0]
    Output: (see reference implementation for expected output)

## Constraints / 约束

- 1 ≤ sample_size ≤ 1000000 / `sample_size` 的范围为 1 至 1,000,000
- 1 ≤ k ≤ 1000 / `k` 的范围为 1 至 1000
- All arrays are float32 except labels, which is int32 / 除 `labels` 为 int32 外，所有数组均为 float32
- Performance is measured with `k` = 5, `max_iterations` = 30, `sample_size` = 10,000 / 性能测试使用 `k` = 5、`max_iterations` = 30、`sample_size` = 10,000
