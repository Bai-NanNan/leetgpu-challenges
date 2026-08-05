Implement batch normalization forward pass for 2D input tensors. Given an input tensor of shape \[N, C\] where N is the batch size and C is the number of features, compute the normalized output using learnable scale (`gamma`) and shift (`beta`) parameters.

For each feature channel j, batch normalization computes: 

$$
\begin{align}
  \mu_j &= \frac{1}{N} \sum_{i=1}^{N} x_{i,j} \\
  \sigma_j^2 &= \frac{1}{N} \sum_{i=1}^{N} (x_{i,j} - \mu_j)^2 \\
  \hat{x}_{i,j} &= \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} \\
  y_{i,j} &= \gamma_j \hat{x}_{i,j} + \beta_j
  \end{align}
$$

对每个特征通道，在 batch 维度上计算均值和方差，进行归一化后使用 `gamma` 缩放、`beta` 平移，最终得到输出张量。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` tensor

## Example 1 / 示例 1:

    Input:  input = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  (N=3, C=2)
            gamma = [1.0, 1.0]
            beta = [0.0, 0.0]
            eps = 1e-5
    Output: output = [[-1.224, -1.224], [0.0, 0.0], [1.224, 1.224]]

## Example 2 / 示例 2:

    Input:  input = [[0.0, 1.0], [2.0, 3.0]]  (N=2, C=2)
            gamma = [2.0, 0.5]
            beta = [1.0, -1.0]
            eps = 1e-5
    Output: output = [[-1.0, -1.5], [3.0, -0.5]]

## Constraints / 约束

- 1 ≤ `N` ≤ 10,000
- 1 ≤ `C` ≤ 1,024
- `eps` = 1e-5
- -100.0 ≤ input values ≤ 100.0
- 0.1 ≤ gamma values ≤ 10.0
- -10.0 ≤ beta values ≤ 10.0
- Performance is measured with `N` = 5,000
