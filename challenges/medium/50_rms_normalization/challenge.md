Implement RMS Normalization forward pass for 1D input vectors. Given an input tensor of shape \[N\] where N is the number of elements, compute the normalized output using a scalar scale (`gamma`) and shift (`beta`) parameter.

RMS Normalization computes: 

$$
\begin{align}
  \text{rms} &= \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon} \\
  \hat{x}_i &= \frac{x_i}{\text{rms}} \\
  y_i &= \gamma \hat{x}_i + \beta
  \end{align}
$$

先计算输入向量的均方根，再用该 RMS 对每个元素归一化，最后应用标量 `gamma` 和 `beta` 得到输出。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` tensor

## Example 1 / 示例 1:

    Input:  input = [1.0, 2.0, 3.0, 4.0]  (N=4)
            gamma = 1.0
            beta = 0.0
            eps = 1e-5
    Output: output = [0.36514813, 0.73029625, 1.0954444, 1.4605925 ]

## Example 2 / 示例 2:

    Input:  input = [1.0, 2.0, 3.0]  (N=3)
            gamma = 1.0
            beta = 0.0
            eps = 1e-5
    Output: output = [0.46290955, 0.9258191, 1.3887286]

## Constraints / 约束

- 1 ≤ `N` ≤ 100,000
- `eps` = 1e-5
- -100.0 ≤ input values ≤ 100.0
- 0.1 ≤ gamma ≤ 10.0
- -10.0 ≤ beta ≤ 10.0
- Performance is measured with `N` = 100,000
