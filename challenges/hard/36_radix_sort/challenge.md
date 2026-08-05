Implement a radix sort algorithm that sorts an array of 32-bit unsigned integers on a GPU. The program should take an input array of unsigned integers and sort them in ascending order using the radix sort algorithm. The `input` parameter contains the unsorted array, and the sorted result should be stored in the `output` array.

实现一个在 GPU 上对 32 位无符号整数数组进行排序的基数排序算法。程序应接收无符号整数输入数组，并使用基数排序按升序排列。`input` 参数包含未排序数组，排序结果应存储在 `output` 数组中。

## Implementation Requirements / 实现要求

- External libraries are not permitted / 不得使用外部库
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final sorted result must be stored in the `output` array / 最终排序结果必须存储在 `output` 数组中
- Use radix sort algorithm (not other sorting algorithms) / 必须使用基数排序算法，不得使用其他排序算法
- Sort in ascending order / 按升序排序

## Example 1 / 示例 1

      Input:  [170, 45, 75, 90, 2, 802, 24, 66]
      Output: [2, 24, 45, 66, 75, 90, 170, 802]
      

## Example 2 / 示例 2

      Input:  [1, 4, 1, 3, 555, 1000, 2]
      Output: [1, 1, 2, 3, 4, 555, 1000]
      

## Constraints / 约束

- `1 ≤ N ≤ 100,000,000` / `N` 的范围为 1 至 100,000,000
- `0 ≤ input[i] ≤ 4,294,967,295` (32-bit unsigned integers) / `input[i]` 的范围为 0 至 4,294,967,295（32 位无符号整数）
- Performance is measured with `N` = 50,000,000 / 性能测试使用 `N` = 50,000,000
