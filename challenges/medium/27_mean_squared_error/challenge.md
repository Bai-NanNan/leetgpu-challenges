Implement a GPU program to calculate the Mean Squared Error (MSE) between predicted values and target values. Given two arrays of equal length, `predictions` and `targets`, compute: 

$$
\text{MSE} =
  \frac{1}{N} \sum_{i=1}^{N} (predictions_i - targets_i)^2
$$

 where N is the number of elements in each array.

均方误差是预测值与目标值之差的平方的平均值。两个输入数组长度相同，最终标量结果应存储在 `mse` 中。

## Implementation Requirements / 实现要求

- External libraries are not permitted.
- The `solve` function signature must remain unchanged.
- The final result must be stored in the `mse` variable.

## Example 1 / 示例 1:

      Input:  predictions = [1.0, 2.0, 3.0, 4.0]
              targets = [1.5, 2.5, 3.5, 4.5]
      Output: mse = 0.25

## Example 2 / 示例 2:

      Input:  predictions = [10.0, 20.0, 30.0]
              targets = [12.0, 18.0, 33.0]
      Output: mse = 5.67

## Constraints / 约束

- 1 ≤ `N` ≤ 100,000,000
- -1000.0 ≤ `predictions[i]`, `targets[i]` ≤ 1000.0
- Performance is measured with `N` = 50,000,000
