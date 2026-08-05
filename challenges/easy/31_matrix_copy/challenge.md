Implement a program that copies an $N \times N$ matrix of 32-bit floating point numbers from input array $A$ to output array $B$ on the GPU. The program should perform a direct element-wise copy so that $B_{i,j} = A_{i,j}$ for all valid indices.

编写一个 GPU 程序，将由 32 位浮点数组成的 $N \times N$ 矩阵从输入数组 $A$ 复制到输出数组 $B$。程序应执行直接的逐元素复制，使所有有效索引都满足 $B_{i,j} = A_{i,j}$。

## Implementation Requirements / 实现要求

- External libraries are not permitted / 不得使用外部库
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in matrix `B` / 最终结果必须存储在矩阵 `B` 中

## Example 1 / 示例 1

    Input:  A = [[1.0, 2.0],
                 [3.0, 4.0]]
    Output: B = [[1.0, 2.0],
                 [3.0, 4.0]]

## Example 2 / 示例 2

    Input:  A = [[5.5, 6.6, 7.7],
                 [8.8, 9.9, 10.1],
                 [11.2, 12.3, 13.4]]
    Output: B = [[5.5, 6.6, 7.7],
                 [8.8, 9.9, 10.1],
                 [11.2, 12.3, 13.4]]

## Constraints / 约束

- 1 ≤ `N` ≤ 4096 / `N` 的范围为 1 至 4096
- All elements are 32-bit floating point numbers / 所有元素均为 32 位浮点数
- Performance is measured with `N` = 4,096 / 性能测试使用 `N` = 4,096
