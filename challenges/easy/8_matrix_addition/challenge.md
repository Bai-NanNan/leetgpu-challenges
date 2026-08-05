Implement a program that performs element-wise addition of two $N \times N$ matrices containing 32-bit floating point numbers on a GPU. The program should take two input matrices of equal dimensions and produce a single output matrix containing their element-wise sum.

编写一个 GPU 程序，对两个包含 32 位浮点数的 $N \times N$ 矩阵执行逐元素加法。程序应接收两个尺寸相同的输入矩阵，并生成一个包含它们逐元素之和的输出矩阵。

## Implementation Requirements / 实现要求

- External libraries are not permitted / 不得使用外部库
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in matrix `C` / 最终结果必须存储在矩阵 `C` 中

## Example 1 / 示例 1

    Input:  A = [[1.0, 2.0],
                 [3.0, 4.0]]
            B = [[5.0, 6.0],
                 [7.0, 8.0]]
    Output: C = [[6.0, 8.0],
                 [10.0, 12.0]]

## Example 2 / 示例 2

    Input:  A = [[1.5, 2.5, 3.5],
                 [4.5, 5.5, 6.5],
                 [7.5, 8.5, 9.5]]
            B = [[0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5]]
    Output: C = [[2.0, 3.0, 4.0],
                 [5.0, 6.0, 7.0],
                 [8.0, 9.0, 10.0]]

## Constraints / 约束

- Input matrices `A` and `B` have identical dimensions / 输入矩阵 `A` 和 `B` 的尺寸相同
- 1 ≤ `N` ≤ 4096 / `N` 的范围为 1 至 4096
- All elements are 32-bit floating point numbers / 所有元素均为 32 位浮点数
- Performance is measured with `N` = 4,096 / 性能测试使用 `N` = 4,096
