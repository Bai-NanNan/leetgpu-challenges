Implement a GPU program that performs clipping on 1D input vectors. Given an input tensor of shape \[N\] where N is the number of elements, compute the output by clipping each element to a specified range \[`lo`, `hi`\]. The input and output tensor must be of type `float32`.

编写一个 GPU 程序，对一维输入向量执行裁剪。给定形状为 \[N\] 的输入张量（其中 N 为元素数量），将每个元素裁剪到指定范围 \[`lo`, `hi`\] 内以计算输出。输入和输出张量必须为 `float32` 类型。

Clipping is defined as:

裁剪定义如下：

1.  For each element `x` in the input tensor, "clip" the element so that it falls within the allowed range `[lo, hi]`. / 对输入张量中的每个元素 `x` 进行“裁剪”，使其落在允许范围 `[lo, hi]` 内。
2.  This operation ensures all values are within the specified range and is commonly used in ML for activation stabilization and pre-quantization. / 此操作确保所有值都在指定范围内，常用于机器学习中的激活稳定化和量化前处理。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted) / 只能使用原生功能（不得使用外部库）
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in the `output` tensor / 最终结果必须存储在 `output` 张量中

## Example 1 / 示例 1

    Input:  [1.5, -2.0, 3.0, 4.5], lo = 0.0, hi = 3.5
    Output: [1.5, 0.0, 3.0, 3.5]

## Example 2 / 示例 2

    Input:  [-1.0, 2.0, 5.0], lo = -0.5, hi = 2.5
    Output: [-0.5, 2.0, 2.5]

## Constraints / 约束

- 1 ≤ `N` ≤ 100,000 / `N` 的范围为 1 至 100,000
- -10<sup>6</sup> ≤ input\[i\] ≤ 10<sup>6</sup> / `input[i]` 的范围为 -10<sup>6</sup> 至 10<sup>6</sup>
- `lo` ≤ `hi` / `lo` 不大于 `hi`
- Performance is measured with `N` = 100,000 / 性能测试使用 `N` = 100,000
