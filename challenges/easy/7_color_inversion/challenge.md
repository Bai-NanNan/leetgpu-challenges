Write a program to invert the colors of an image. The image is represented as a 1D array of RGBA (Red, Green, Blue, Alpha) values, where each component is an 8-bit unsigned integer (`unsigned char`).

编写一个程序，对图像执行颜色反转。图像表示为 RGBA（红、绿、蓝、透明度）值组成的一维数组，其中每个分量都是 8 位无符号整数（`unsigned char`）。

Color inversion is performed by subtracting each color component (R, G, B) from 255. The Alpha component should remain unchanged.

颜色反转的方式是用 255 减去每个颜色分量（R、G、B）。Alpha 分量应保持不变。

The input array `image` will contain `width * height * 4` elements. The first 4 elements represent the RGBA values of the top-left pixel, the next 4 elements represent the pixel to its right, and so on.

输入数组 `image` 包含 `width * height * 4` 个元素。前 4 个元素表示左上角像素的 RGBA 值，接下来的 4 个元素表示其右侧像素的值，依此类推。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted) / 只能使用原生功能（不得使用外部库）
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in the array `image` / 最终结果必须存储在数组 `image` 中

## Example 1 / 示例 1

    Input: image = [255, 0, 128, 255, 0, 255, 0, 255], width=1, height=2
    Output: [0, 255, 127, 255, 255, 0, 255, 255]

## Example 2 / 示例 2

    Input: image = [10, 20, 30, 255, 100, 150, 200, 255], width=2, height=1
    Output: [245, 235, 225, 255, 155, 105, 55, 255]

## Constraints / 约束

- 1 ≤ `width` ≤ 4096 / `width` 的范围为 1 至 4096
- 1 ≤ `height` ≤ 4096 / `height` 的范围为 1 至 4096
- `width * height` ≤ 8,388,608. / `width * height` 不超过 8,388,608。
- Performance is measured with `height` = 5,120, `width` = 4,096 / 性能测试使用 `height` = 5,120、`width` = 4,096
