Run batched autoregressive inference for a 10-parameter transformer that adds two 10-digit numbers. Given prompts of shape `[batch_size, 31]` (int32) and a 10-float weight buffer, write output logits of shape `[batch_size, 11, 10]` — one logit row per decode step over the 10-digit vocabulary (0–9). All tensors are float32 except the int32 prompts.

The model comes from the [AdderBoard](https://gist.github.com/Lokimorty/d54e5c61997e00fb922b6692739a0f6c) competition for the smallest autoregressive transformer that adds 10-digit numbers at ≥99% accuracy. It encodes carry propagation in 10 learned parameters via RoPE geometry, tied embeddings, and SwiGLU gating.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgNzIwIDU0MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiBzdHlsZT0iZGlzcGxheTpibG9jazsgbWFyZ2luOjIwcHggYXV0bzsgbWF4LXdpZHRoOjcyMHB4OyIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMyI+CiAgPHJlY3Qgd2lkdGg9IjcyMCIgaGVpZ2h0PSI1NDAiIHJ4PSIxMiIgZmlsbD0iIzIyMiIgLz4KCiAgPCEtLSBJbnB1dCAtLT4KICA8cmVjdCB4PSIyNzAiIHk9IjIwIiB3aWR0aD0iMTgwIiBoZWlnaHQ9IjM2IiByeD0iNiIgZmlsbD0iIzMzNSIgc3Ryb2tlPSIjNDQ3N2JiIiAvPgogIDx0ZXh0IHg9IjM2MCIgeT0iNDMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNjY2MiPlRva2VuIFByb21wdCBbQiwzMV08L3RleHQ+CgogIDwhLS0gRW1iZWRkaW5nIC0tPgogIDxyZWN0IHg9IjI1MCIgeT0iODAiIHdpZHRoPSIyMjAiIGhlaWdodD0iMzYiIHJ4PSI2IiBmaWxsPSIjMmE0YTJhIiBzdHJva2U9IiM0NGFhNjYiIC8+CiAgPHRleHQgeD0iMzYwIiB5PSIxMDMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNjY2MiPkVtYmVkOiBbdzAtdzEqZMKyLCAtZF08L3RleHQ+CiAgPGxpbmUgeDE9IjM2MCIgeTE9IjU2IiB4Mj0iMzYwIiB5Mj0iODAiIHN0cm9rZT0iIzY2NiIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYXJyKSI+PC9saW5lPgoKICA8IS0tIFVuaXQgUk1TTm9ybSAxIC0tPgogIDxyZWN0IHg9IjI3MCIgeT0iMTQwIiB3aWR0aD0iMTgwIiBoZWlnaHQ9IjMyIiByeD0iNiIgZmlsbD0iIzMzMyIgc3Ryb2tlPSIjODg4IiAvPgogIDx0ZXh0IHg9IjM2MCIgeT0iMTYxIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjY2NjIj5Vbml0IFJNU05vcm08L3RleHQ+CiAgPGxpbmUgeDE9IjM2MCIgeTE9IjExNiIgeDI9IjM2MCIgeTI9IjE0MCIgc3Ryb2tlPSIjNjY2IiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhcnIpIj48L2xpbmU+CgogIDwhLS0gQXR0ZW50aW9uIGJsb2NrIC0tPgogIDxyZWN0IHg9IjIwMCIgeT0iMTk1IiB3aWR0aD0iMzIwIiBoZWlnaHQ9IjEwNSIgcng9IjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzQ0NzdiYiIgc3Ryb2tlLWRhc2hhcnJheT0iNCIgLz4KICA8dGV4dCB4PSIyMTAiIHk9IjIxMyIgZmlsbD0iIzQ0NzdiYiIgZm9udC1zaXplPSIxMSI+U2VsZi1BdHRlbnRpb24gKDEgaGVhZCwgZGltPTIpPC90ZXh0PgoKICA8cmVjdCB4PSIyMTUiIHk9IjIyMCIgd2lkdGg9IjkwIiBoZWlnaHQ9IjI4IiByeD0iNCIgZmlsbD0iIzMzNSIgc3Ryb2tlPSIjNDQ3N2JiIiAvPgogIDx0ZXh0IHg9IjI2MCIgeT0iMjM5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjExIj5RIFByb2ogWzJwXTwvdGV4dD4KICA8cmVjdCB4PSIzMTUiIHk9IjIyMCIgd2lkdGg9IjkwIiBoZWlnaHQ9IjI4IiByeD0iNCIgZmlsbD0iIzMzNSIgc3Ryb2tlPSIjNDQ3N2JiIiAvPgogIDx0ZXh0IHg9IjM2MCIgeT0iMjM5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjExIj5LIFByb2ogWzBwXTwvdGV4dD4KICA8cmVjdCB4PSI0MTUiIHk9IjIyMCIgd2lkdGg9IjkwIiBoZWlnaHQ9IjI4IiByeD0iNCIgZmlsbD0iIzMzNSIgc3Ryb2tlPSIjNDQ3N2JiIiAvPgogIDx0ZXh0IHg9IjQ2MCIgeT0iMjM5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjExIj5WIFByb2ogWzFwXTwvdGV4dD4KCiAgPHJlY3QgeD0iMjE1IiB5PSIyNTgiIHdpZHRoPSIyOTAiIGhlaWdodD0iMjgiIHJ4PSI0IiBmaWxsPSIjMzM1IiBzdHJva2U9IiM0NDc3YmIiIC8+CiAgPHRleHQgeD0iMzYwIiB5PSIyNzciIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNjY2MiIGZvbnQtc2l6ZT0iMTEiPlFLIE5vcm0gKyBSb1BFKM+JPTLPgC8xOSkgKyBDYXVzYWwgQXR0bjwvdGV4dD4KCiAgPGxpbmUgeDE9IjM2MCIgeTE9IjE3MiIgeDI9IjM2MCIgeTI9IjE5NSIgc3Ryb2tlPSIjNjY2IiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhcnIpIj48L2xpbmU+CgogIDwhLS0gUmVzaWR1YWwgMSAtLT4KICA8dGV4dCB4PSI1NTUiIHk9IjI2NSIgZmlsbD0iIzg4OCIgZm9udC1zaXplPSIxMSI+KyByZXNpZHVhbDwvdGV4dD4KICA8bGluZSB4MT0iNTQwIiB5MT0iOTgiIHgyPSI1NzAiIHkyPSI5OCIgc3Ryb2tlPSIjODg4IiBzdHJva2Utd2lkdGg9IjEiIHN0cm9rZS1kYXNoYXJyYXk9IjMiPjwvbGluZT4KICA8bGluZSB4MT0iNTcwIiB5MT0iOTgiIHgyPSI1NzAiIHkyPSIzMjAiIHN0cm9rZT0iIzg4OCIgc3Ryb2tlLXdpZHRoPSIxIiBzdHJva2UtZGFzaGFycmF5PSIzIj48L2xpbmU+CiAgPGxpbmUgeDE9IjU3MCIgeTE9IjMyMCIgeDI9IjUyMCIgeTI9IjMyMCIgc3Ryb2tlPSIjODg4IiBzdHJva2Utd2lkdGg9IjEiIHN0cm9rZS1kYXNoYXJyYXk9IjMiIG1hcmtlci1lbmQ9InVybCgjYXJyKSI+PC9saW5lPgoKICA8IS0tIEFkZCBub2RlIDEgLS0+CiAgPGNpcmNsZSBjeD0iNTAwIiBjeT0iMzIwIiByPSIxNCIgZmlsbD0iIzMzMyIgc3Ryb2tlPSIjODg4Ij48L2NpcmNsZT4KICA8dGV4dCB4PSI1MDAiIHk9IjMyNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2NjYyIgZm9udC1zaXplPSIxNiI+KzwvdGV4dD4KICA8bGluZSB4MT0iMzYwIiB5MT0iMzAwIiB4Mj0iMzYwIiB5Mj0iMzIwIiBzdHJva2U9IiM2NjYiIHN0cm9rZS13aWR0aD0iMS41Ij48L2xpbmU+CiAgPGxpbmUgeDE9IjM2MCIgeTE9IjMyMCIgeDI9IjQ4NiIgeTI9IjMyMCIgc3Ryb2tlPSIjNjY2IiBzdHJva2Utd2lkdGg9IjEuNSI+PC9saW5lPgoKICA8IS0tIFVuaXQgUk1TTm9ybSAyIC0tPgogIDxyZWN0IHg9IjI3MCIgeT0iMzUwIiB3aWR0aD0iMTgwIiBoZWlnaHQ9IjMyIiByeD0iNiIgZmlsbD0iIzMzMyIgc3Ryb2tlPSIjODg4IiAvPgogIDx0ZXh0IHg9IjM2MCIgeT0iMzcxIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjY2NjIj5Vbml0IFJNU05vcm08L3RleHQ+CiAgPGxpbmUgeDE9IjUwMCIgeTE9IjMzNCIgeDI9IjUwMCIgeTI9IjM0MiIgc3Ryb2tlPSIjNjY2IiBzdHJva2Utd2lkdGg9IjEuNSI+PC9saW5lPgogIDxsaW5lIHgxPSI1MDAiIHkxPSIzNDIiIHgyPSIzNjAiIHkyPSIzNDIiIHN0cm9rZT0iIzY2NiIgc3Ryb2tlLXdpZHRoPSIxLjUiPjwvbGluZT4KICA8bGluZSB4MT0iMzYwIiB5MT0iMzQyIiB4Mj0iMzYwIiB5Mj0iMzUwIiBzdHJva2U9IiM2NjYiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiPjwvbGluZT4KCiAgPCEtLSBNTFAgYmxvY2sgLS0+CiAgPHJlY3QgeD0iMjAwIiB5PSI0MDAiIHdpZHRoPSIzMjAiIGhlaWdodD0iMzYiIHJ4PSI2IiBmaWxsPSIjMmE0YTJhIiBzdHJva2U9IiM0NGFhNjYiIC8+CiAgPHRleHQgeD0iMzYwIiB5PSI0MjMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNjY2MiIGZvbnQtc2l6ZT0iMTIiPk1MUDogR2F0ZSArIFN3aUdMVSArIENhcnJ5IFszcF08L3RleHQ+CiAgPGxpbmUgeDE9IjM2MCIgeTE9IjM4MiIgeDI9IjM2MCIgeTI9IjQwMCIgc3Ryb2tlPSIjNjY2IiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhcnIpIj48L2xpbmU+CgogIDwhLS0gRmluYWwgbm9ybSArIG91dHB1dCAtLT4KICA8cmVjdCB4PSIyNTAiIHk9IjQ2MCIgd2lkdGg9IjIyMCIgaGVpZ2h0PSIzNiIgcng9IjYiIGZpbGw9IiMzMzMiIHN0cm9rZT0iIzg4OCIgLz4KICA8dGV4dCB4PSIzNjAiIHk9IjQ4MyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2NjYyI+Uk1TTm9ybSBbMnBdICsgTG9naXRzPC90ZXh0PgogIDxsaW5lIHgxPSIzNjAiIHkxPSI0MzYiIHgyPSIzNjAiIHkyPSI0NjAiIHN0cm9rZT0iIzY2NiIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYXJyKSI+PC9saW5lPgoKICA8IS0tIFBhcmFtIGNvdW50cyAtLT4KICA8dGV4dCB4PSIzMCIgeT0iNTIwIiBmaWxsPSIjNjY2IiBmb250LXNpemU9IjExIj5Ub3RhbDogMTAgcGFyYW1ldGVycyAoMisyKzErMisxKzIpPC90ZXh0PgoKICA8ZGVmcz4KICAgIDxtYXJrZXIgaWQ9ImFyciIgbWFya2Vyd2lkdGg9IjgiIG1hcmtlcmhlaWdodD0iNiIgcmVmeD0iOCIgcmVmeT0iMyIgb3JpZW50PSJhdXRvIj4KICAgICAgPHBhdGggZD0iTTAsMCBMOCwzIEwwLDYiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzY2NiIgc3Ryb2tlLXdpZHRoPSIxIiAvPgogICAgPC9tYXJrZXI+CiAgPC9kZWZzPgo8L3N2Zz4=)

中文说明：该题要求对一批 31-token 的数字加法 prompt 执行 11 步自回归推理。模型是一个隐藏维度为 2 的单层 Transformer，使用 RoPE、SwiGLU、RMSNorm 和权重绑定，并将每一步最后位置的 logits 写入输出。

## Model Architecture / 模型架构

Single-layer pre-norm transformer. Hidden dim 2, 1 head, head dim 2, vocab 10 (digits 0–9), tied input/output embeddings.

Each step runs the full sequence `[batch_size, seq_len, 2]` through:

**1. Token Embedding** (2 parameters: `w0`, `w1`)

$$
e(d) = \begin{bmatrix} w_0 - w_1 \cdot d^2 \\ -d \end{bmatrix}
$$

**2. Unit RMSNorm** (no parameters)

$$
\text{UnitRMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}, \quad \epsilon = 10^{-6}
$$

**3. Self-Attention** (3 parameters: `q0`, `q1`, `v0`)

Projections applied to the normed hidden state `h` with shape `[*, 2]`:

$$
Q = \begin{bmatrix} h_0 \cdot q_0 \\ h_0 \cdot q_1 \end{bmatrix}, \quad
K = \begin{bmatrix} h_0 \\ 0 \end{bmatrix}, \quad
V = \begin{bmatrix} h_1 \cdot v_0 \\ 0 \end{bmatrix}
$$

After projection, Q and K are each normalized with Unit RMSNorm, then RoPE is applied with angular frequency `ω = 2π/19`:

$$
\text{RoPE}(x, p) = \begin{bmatrix} x_0 \cos(p\omega) - x_1 \sin(p\omega) \\
x_0 \sin(p\omega) + x_1 \cos(p\omega) \end{bmatrix}
$$

Scaled dot-product attention with causal mask uses scale factor:

$$
\text{scale} = \frac{1}{\sqrt{d_h}} \cdot S^2
$$

where $d_h = 2$ is the head dimension and $S^2$ is the QK-norm scale constant (see weight table below for exact value).

The output projection maps `[attn_0, attn_1]` → `[0, attn_0]` (no parameters), followed by a residual connection.

**4. MLP** (3 parameters: `a`, `c`, `carry`)

Applied to the unit-RMSNorm of the post-attention hidden state:

$$
g_0 = h_0 \cdot a + h_1 \cdot c, \quad g_1 = h_0 \cdot (a - c / 1000) + h_1 \cdot c
$$

$$
\text{base} = h_0, \quad \text{up} = [\text{base}, \text{base}]
$$

$$
\text{mix} = \text{SiLU}([g_0, g_1]) \odot \text{up}
$$

$$
\text{MLP}(h) = \begin{bmatrix} 0 \\ \text{carry} \cdot (\text{mix}_1 - \text{mix}_0) \end{bmatrix}
$$

followed by a residual connection.

**5. Final RMSNorm** (2 parameters: `n0`, `n1`)

Standard RMSNorm with learned weight:

$$
\text{out} = \frac{h}{\sqrt{\text{mean}(h^2) + \epsilon}} \odot [n_0, n_1]
$$

**6. Output Logits** (tied with embedding)

$$
\text{logits} = \text{out} \cdot E^T \quad \text{where } E_{d} = e(d)
$$

## Autoregressive Decoding / 自回归解码

Starting from the 31-token prompt, repeat 11 times:

1.  Run the full forward pass on the current sequence
2.  Extract logits at the last position → store in output
3.  Append `argmax(logits)` as the next token

The sequence grows from length 31 to 42 over the 11 decode steps.

## Weight Layout / 权重布局

| Offset | Size | Name   | Description                          |
|--------|------|--------|--------------------------------------|
| 0      | 2    | embed  | Embedding: `e(d) = [w0 - w1*d², -d]` |
| 2      | 2    | q_proj | Q projection weights `[q0, q1]`      |
| 4      | 1    | v_proj | V projection weight `v0`             |
| 5      | 2    | gate   | MLP gate weights `[a, c]`            |
| 7      | 1    | carry  | MLP carry weight                     |
| 8      | 2    | norm   | Final RMSNorm weight `[n0, n1]`      |

## Token Encoding / Token 编码

Each input pair `(a, b)` of 10-digit numbers is encoded as a 31-token sequence:

    [0, a_rev_0, ..., a_rev_9, 0, 0, 0, 0, 0, 0, 0, 0, 0, b_rev_0, ..., b_rev_9, 0]

where `a_rev` and `b_rev` are the digits in least-significant-first order, zero-padded to 10 digits. The model then generates 11 output tokens (digits of the sum, also least-significant-first).

## Implementation Requirements / 实现要求

- Implement `solve(prompts, output, weights, batch_size)` with the exact signature shown (JAX exception: `solve(prompts, weights, batch_size)` returns the output tensor directly)
- Do not use any external libraries beyond what the framework provides
- The function must write logits into the `output` buffer (except JAX, which returns it)
- Architecture constants are fixed: `vocab_size` = 10, `hidden_dim` = 2, `head_dim` = 2, `num_heads` = 1, `prompt_len` = 31, `decode_steps` = 11
- RMSNorm epsilon = 10<sup>−6</sup>
- RoPE angular frequency ω = 2π/19
- Attention scale = (1/√2) · `S`² where `S`² = ln(10) / (√2 · (cos(0.3ω) − cos(0.7ω)))
- SiLU activation: `silu(x) = x · sigmoid(x)`

## Example / 示例

With `batch_size` = 2 and pairs (3, 5), (99, 1):

    Input prompts (shape [2, 31]):
      [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      [0, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    Output logits shape: [2, 11, 10]
      (logits at each of 11 decode steps over 10 digit classes)

    Expected decoded tokens (via argmax):
      Pair (3, 5):   sum = 8       → [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      Pair (99, 1):  sum = 100     → [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

## Constraints / 约束

- `batch_size`: 1 ≤ `batch_size` ≤ 100,000
- `prompts`: 32-bit integer tensor, values in \[0, 9\]
- `weights`: 32-bit float tensor with exactly 10 elements
- `output`: 32-bit float tensor of shape `[batch_size, 11, 10]`
- Input numbers are in range \[0, 9,999,999,999\] (10-digit unsigned integers)
- Performance is measured with `batch_size` = 100,000
