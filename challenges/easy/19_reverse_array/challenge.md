Implement a program that reverses an array of 32-bit floating point numbers in-place. The program should perform an in-place reversal of `input`.

编写一个程序，就地反转由 32 位浮点数组成的数组。程序应对 `input` 执行原地反转。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted) / 只能使用原生功能（不得使用外部库）
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored back in `input` / 最终结果必须写回 `input`

## Example 1 / 示例 1

    Input: [1.0, 2.0, 3.0, 4.0]
    Output: [4.0, 3.0, 2.0, 1.0]

## Example 2 / 示例 2

    Input: [1.5, 2.5, 3.5]
    Output: [3.5, 2.5, 1.5]

## Constraints / 约束

- 1 ≤ `N` ≤ 100,000,000 / `N` 的范围为 1 至 100,000,000
- Performance is measured with `N` = 25,000,000 / 性能测试使用 `N` = 25,000,000
