Implement a GPU program that converts an RGB image to grayscale on the GPU. Given an input RGB image represented as a 1D array of 32-bit floating point values, compute the corresponding grayscale image using the standard RGB to grayscale conversion formula.

编写一个 GPU 程序，在 GPU 上将 RGB 图像转换为灰度图像。给定由 32 位浮点数组成的一维数组表示的 RGB 输入图像，使用标准的 RGB 转灰度公式计算对应的灰度图像。

The conversion formula is: `gray = 0.299 × R + 0.587 × G + 0.114 × B`

转换公式为：`gray = 0.299 × R + 0.587 × G + 0.114 × B`

The input array `input` contains `height × width × 3` elements, where the RGB values for each pixel are stored consecutively (R, G, B, R, G, B, ...). The output array `output` should contain `height × width` grayscale values.

输入数组 `input` 包含 `height × width × 3` 个元素，每个像素的 RGB 值连续存储（R、G、B、R、G、B、……）。输出数组 `output` 应包含 `height × width` 个灰度值。

## Implementation Requirements / 实现要求

- External libraries are not permitted / 不得使用外部库
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in the array `output` / 最终结果必须存储在数组 `output` 中
- Use the exact coefficients: 0.299 for red, 0.587 for green, 0.114 for blue / 必须使用精确系数：红色为 0.299、绿色为 0.587、蓝色为 0.114

## Example 1 / 示例 1

    Input:  input = [255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 128.0, 128.0, 128.0], width=2, height=2
    Output: output = [76.245, 149.685, 29.07, 128.0]

## Example 2 / 示例 2

    Input:  input = [100.0, 150.0, 200.0], width=1, height=1
    Output: output = [140.75]

## Constraints / 约束

- 1 ≤ `width` ≤ 4096 / `width` 的范围为 1 至 4096
- 1 ≤ `height` ≤ 4096 / `height` 的范围为 1 至 4096
- `width × height` ≤ 4,194,304 / `width × height` 不超过 4,194,304
- All RGB values are in the range \[0.0, 255.0\] / 所有 RGB 值均在 \[0.0, 255.0\] 范围内
- Performance is measured with `height` = 2,048, `width` = 2,048 / 性能测试使用 `height` = 2,048、`width` = 2,048
