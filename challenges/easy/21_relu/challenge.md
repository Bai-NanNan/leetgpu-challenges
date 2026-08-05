Implement a program that performs the Rectified Linear Unit (ReLU) activation function on a vector of 32-bit floating point numbers. The ReLU function sets all negative values to zero and leaves positive values unchanged: 

编写一个程序，对由 32 位浮点数组成的向量执行修正线性单元（ReLU）激活函数。ReLU 函数将所有负值设为零，并保持正值不变：

$$
\text{ReLU}(x) = \max(0, x)
$$

## Implementation Requirements / 实现要求

- External libraries are not permitted / 不得使用外部库
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in `output` / 最终结果必须存储在 `output` 中

## Example 1 / 示例 1

    Input:  input = [-2.0, -1.0, 0.0, 1.0, 2.0]
    Output: output = [0.0, 0.0, 0.0, 1.0, 2.0]

## Example 2 / 示例 2

    Input:  input = [-3.5, 0.0, 4.2]
    Output: output = [0.0, 0.0, 4.2]

## Constraints / 约束

- 1 ≤ `N` ≤ 100,000,000 / `N` 的范围为 1 至 100,000,000
- Performance is measured with `N` = 25,000,000 / 性能测试使用 `N` = 25,000,000
