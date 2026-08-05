Run inference on a PyTorch model. Given an input tensor and a trained `torch.nn.Linear` model, compute the forward pass and store the result in the output tensor.

对 PyTorch 模型执行推理。给定一个输入张量和训练好的 `torch.nn.Linear` 模型，计算前向传播结果并将其存储在输出张量中。

The model performs a linear transformation: `output = input @ weight.T + bias` where `weight` has shape \[output_size, input_size\] and `bias` has shape \[output_size\].

该模型执行线性变换：`output = input @ weight.T + bias`，其中 `weight` 的形状为 \[output_size, input_size\]，`bias` 的形状为 \[output_size\]。

## Implementation Requirements / 实现要求

- Use PyTorch's built-in functions and operations / 使用 PyTorch 内置的函数和运算
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in the `output` tensor / 最终结果必须存储在 `output` 张量中
- The model is already loaded and ready for inference / 模型已经加载完毕，可以直接用于推理

## Example 1 / 示例 1

      Input:  input = [[1.0, 2.0]]  (batch_size=1, input_size=2)
              model: Linear layer with weight=[[0.5, 1.0], [1.5, 0.5]], bias=[0.1, 0.2]
      Output: output = [[2.6, 2.7]]  (batch_size=1, output_size=2)
      

## Example 2 / 示例 2

      Input:  input = [[1.0], [2.0], [3.0]]  (batch_size=3, input_size=1)
              model: Linear layer with weight=[[2.0]], bias=[1.0]
      Output: output = [[3.0], [5.0], [7.0]]  (batch_size=3, output_size=1)
      

## Constraints / 约束

- 1 ≤ `batch_size` ≤ 1,000 / `batch_size` 的范围为 1 至 1,000
- 1 ≤ `input_size` ≤ 1,000 / `input_size` 的范围为 1 至 1,000
- 1 ≤ `output_size` ≤ 1,000 / `output_size` 的范围为 1 至 1,000
- -10.0 ≤ input values ≤ 10.0 / 输入值的范围为 -10.0 至 10.0
- Performance is measured with `batch_size` = 1,000 / 性能测试使用 `batch_size` = 1,000
