Implement a single GPT-2 transformer decoder block. Given an input tensor $x$ of shape `(seq_len, 768)` and a packed weight buffer containing all block parameters, compute the output using pre-norm architecture with multi-head self-attention and a feed-forward network with GELU activation.

![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdib3g9IjAgMCAzMjAgNjAwIiB3aWR0aD0iMzIwIiBoZWlnaHQ9IjYwMCIgc3R5bGU9ImRpc3BsYXk6YmxvY2s7IG1hcmdpbjoyMHB4IGF1dG87Ij4KICA8ZGVmcz4KICAgIDxtYXJrZXIgaWQ9ImFoIiB2aWV3Ym94PSIwIDAgMTAgMTAiIHJlZng9IjkiIHJlZnk9IjUiIG1hcmtlcndpZHRoPSI2IiBtYXJrZXJoZWlnaHQ9IjYiIG9yaWVudD0iYXV0by1zdGFydC1yZXZlcnNlIj4KICAgICAgPHBhdGggZD0iTTAgMEwxMCA1TDAgMTB6IiBmaWxsPSIjOTk5IiAvPgogICAgPC9tYXJrZXI+CiAgPC9kZWZzPgoKICA8IS0tIElucHV0IGxhYmVsIC0tPgogIDx0ZXh0IHg9IjEzMCIgeT0iMTgiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNjY2MiIGZvbnQtc2l6ZT0iMTMiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiPnggKHNlcV9sZW4sIDc2OCk8L3RleHQ+CgogIDwhLS0gQXJyb3c6IGlucHV0IC0+IExOMSAtLT4KICA8bGluZSB4MT0iMTMwIiB5MT0iMjYiIHgyPSIxMzAiIHkyPSI0NCIgc3Ryb2tlPSIjOTk5IiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhaCkiPjwvbGluZT4KCiAgPCEtLSBSZXNpZHVhbCAxOiBmb3JrIHJpZ2h0LCBkb3duLCBiYWNrIGxlZnQgdG8gQWRkMSAtLT4KICA8bGluZSB4MT0iMTMwIiB5MT0iMzMiIHgyPSIyNjAiIHkyPSIzMyIgc3Ryb2tlPSIjNTU1IiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWRhc2hhcnJheT0iNSw0Ij48L2xpbmU+CiAgPGxpbmUgeDE9IjI2MCIgeTE9IjMzIiB4Mj0iMjYwIiB5Mj0iMjcwIiBzdHJva2U9IiM1NTUiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtZGFzaGFycmF5PSI1LDQiPjwvbGluZT4KICA8bGluZSB4MT0iMjYwIiB5MT0iMjcwIiB4Mj0iMTQ1IiB5Mj0iMjcwIiBzdHJva2U9IiM1NTUiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtZGFzaGFycmF5PSI1LDQiIG1hcmtlci1lbmQ9InVybCgjYWgpIj48L2xpbmU+CiAgPHRleHQgeD0iMjY4IiB5PSIxNTUiIGZpbGw9IiM2NjYiIGZvbnQtc2l6ZT0iMTAiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiB0cmFuc2Zvcm09InJvdGF0ZSg5MCwyNjgsMTU1KSI+cmVzaWR1YWw8L3RleHQ+CgogIDwhLS0gTE4xIC0tPgogIDxyZWN0IHg9IjU1IiB5PSI0NyIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIzMCIgcng9IjUiIGZpbGw9IiMzMzMiIHN0cm9rZT0iIzc3NyIgc3Ryb2tlLXdpZHRoPSIxIiAvPgogIDx0ZXh0IHg9IjEzMCIgeT0iNjciIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNjY2MiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIj5MYXllck5vcm0gMTwvdGV4dD4KICA8bGluZSB4MT0iMTMwIiB5MT0iNzciIHgyPSIxMzAiIHkyPSI5NSIgc3Ryb2tlPSIjOTk5IiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhaCkiPjwvbGluZT4KCiAgPCEtLSBRS1YgUHJvaiAtLT4KICA8cmVjdCB4PSI1NSIgeT0iOTgiIHdpZHRoPSIxNTAiIGhlaWdodD0iMzAiIHJ4PSI1IiBmaWxsPSIjMWUyZDRkIiBzdHJva2U9IiM0NDc3YmIiIHN0cm9rZS13aWR0aD0iMSIgLz4KICA8dGV4dCB4PSIxMzAiIHk9IjExOCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2FhY2NlZSIgZm9udC1zaXplPSIxMiIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiPlFLViBQcm9qZWN0aW9uPC90ZXh0PgogIDxsaW5lIHgxPSIxMzAiIHkxPSIxMjgiIHgyPSIxMzAiIHkyPSIxNDYiIHN0cm9rZT0iIzk5OSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYWgpIj48L2xpbmU+CgogIDwhLS0gTUhBIC0tPgogIDxyZWN0IHg9IjU1IiB5PSIxNDkiIHdpZHRoPSIxNTAiIGhlaWdodD0iMzAiIHJ4PSI1IiBmaWxsPSIjMWUyZDRkIiBzdHJva2U9IiM0NDc3YmIiIHN0cm9rZS13aWR0aD0iMSIgLz4KICA8dGV4dCB4PSIxMzAiIHk9IjE2OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2FhY2NlZSIgZm9udC1zaXplPSIxMiIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiPk11bHRpLUhlYWQgQXR0ZW50aW9uPC90ZXh0PgogIDxsaW5lIHgxPSIxMzAiIHkxPSIxNzkiIHgyPSIxMzAiIHkyPSIxOTciIHN0cm9rZT0iIzk5OSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYWgpIj48L2xpbmU+CgogIDwhLS0gQXR0biBPdXQgUHJvaiAtLT4KICA8cmVjdCB4PSI1NSIgeT0iMjAwIiB3aWR0aD0iMTUwIiBoZWlnaHQ9IjMwIiByeD0iNSIgZmlsbD0iIzFlMmQ0ZCIgc3Ryb2tlPSIjNDQ3N2JiIiBzdHJva2Utd2lkdGg9IjEiIC8+CiAgPHRleHQgeD0iMTMwIiB5PSIyMjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNhYWNjZWUiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIj5PdXRwdXQgUHJvamVjdGlvbjwvdGV4dD4KICA8bGluZSB4MT0iMTMwIiB5MT0iMjMwIiB4Mj0iMTMwIiB5Mj0iMjU4IiBzdHJva2U9IiM5OTkiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FoKSI+PC9saW5lPgoKICA8IS0tIEFkZCAxIC0tPgogIDxjaXJjbGUgY3g9IjEzMCIgY3k9IjI3MCIgcj0iMTIiIGZpbGw9IiMyMjIiIHN0cm9rZT0iIzk5OSIgc3Ryb2tlLXdpZHRoPSIxLjUiPjwvY2lyY2xlPgogIDx0ZXh0IHg9IjEzMCIgeT0iMjc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjE1IiBmb250LWZhbWlseT0ic2Fucy1zZXJpZiIgZm9udC13ZWlnaHQ9ImJvbGQiPis8L3RleHQ+CiAgPGxpbmUgeDE9IjEzMCIgeTE9IjI4MiIgeDI9IjEzMCIgeTI9IjMwNiIgc3Ryb2tlPSIjOTk5IiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhaCkiPjwvbGluZT4KCiAgPCEtLSBSZXNpZHVhbCAyOiBmb3JrIHJpZ2h0LCBkb3duLCBiYWNrIGxlZnQgdG8gQWRkMiAtLT4KICA8bGluZSB4MT0iMTMwIiB5MT0iMjkyIiB4Mj0iMjYwIiB5Mj0iMjkyIiBzdHJva2U9IiM1NTUiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtZGFzaGFycmF5PSI1LDQiPjwvbGluZT4KICA8bGluZSB4MT0iMjYwIiB5MT0iMjkyIiB4Mj0iMjYwIiB5Mj0iNTMwIiBzdHJva2U9IiM1NTUiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtZGFzaGFycmF5PSI1LDQiPjwvbGluZT4KICA8bGluZSB4MT0iMjYwIiB5MT0iNTMwIiB4Mj0iMTQ1IiB5Mj0iNTMwIiBzdHJva2U9IiM1NTUiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtZGFzaGFycmF5PSI1LDQiIG1hcmtlci1lbmQ9InVybCgjYWgpIj48L2xpbmU+CiAgPHRleHQgeD0iMjY4IiB5PSI0MTUiIGZpbGw9IiM2NjYiIGZvbnQtc2l6ZT0iMTAiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiB0cmFuc2Zvcm09InJvdGF0ZSg5MCwyNjgsNDE1KSI+cmVzaWR1YWw8L3RleHQ+CgogIDwhLS0gTE4yIC0tPgogIDxyZWN0IHg9IjU1IiB5PSIzMDkiIHdpZHRoPSIxNTAiIGhlaWdodD0iMzAiIHJ4PSI1IiBmaWxsPSIjMzMzIiBzdHJva2U9IiM3NzciIHN0cm9rZS13aWR0aD0iMSIgLz4KICA8dGV4dCB4PSIxMzAiIHk9IjMyOSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2NjYyIgZm9udC1zaXplPSIxMiIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiPkxheWVyTm9ybSAyPC90ZXh0PgogIDxsaW5lIHgxPSIxMzAiIHkxPSIzMzkiIHgyPSIxMzAiIHkyPSIzNTciIHN0cm9rZT0iIzk5OSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYWgpIj48L2xpbmU+CgogIDwhLS0gRkMgLS0+CiAgPHJlY3QgeD0iNTUiIHk9IjM2MCIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIzMCIgcng9IjUiIGZpbGw9IiMxZTNkMmQiIHN0cm9rZT0iIzQ0YWE2NiIgc3Ryb2tlLXdpZHRoPSIxIiAvPgogIDx0ZXh0IHg9IjEzMCIgeT0iMzgwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjYWFlZWJiIiBmb250LXNpemU9IjEyIiBmb250LWZhbWlseT0ic2Fucy1zZXJpZiI+TGluZWFyICg3Njgg4oaSIDMwNzIpPC90ZXh0PgogIDxsaW5lIHgxPSIxMzAiIHkxPSIzOTAiIHgyPSIxMzAiIHkyPSI0MDgiIHN0cm9rZT0iIzk5OSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYWgpIj48L2xpbmU+CgogIDwhLS0gR0VMVSAtLT4KICA8cmVjdCB4PSI1NSIgeT0iNDExIiB3aWR0aD0iMTUwIiBoZWlnaHQ9IjMwIiByeD0iNSIgZmlsbD0iIzFlM2QyZCIgc3Ryb2tlPSIjNDRhYTY2IiBzdHJva2Utd2lkdGg9IjEiIC8+CiAgPHRleHQgeD0iMTMwIiB5PSI0MzEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNhYWVlYmIiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIj5HRUxVPC90ZXh0PgogIDxsaW5lIHgxPSIxMzAiIHkxPSI0NDEiIHgyPSIxMzAiIHkyPSI0NTkiIHN0cm9rZT0iIzk5OSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYWgpIj48L2xpbmU+CgogIDwhLS0gUHJvaiAtLT4KICA8cmVjdCB4PSI1NSIgeT0iNDYyIiB3aWR0aD0iMTUwIiBoZWlnaHQ9IjMwIiByeD0iNSIgZmlsbD0iIzFlM2QyZCIgc3Ryb2tlPSIjNDRhYTY2IiBzdHJva2Utd2lkdGg9IjEiIC8+CiAgPHRleHQgeD0iMTMwIiB5PSI0ODIiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNhYWVlYmIiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIj5MaW5lYXIgKDMwNzIg4oaSIDc2OCk8L3RleHQ+CiAgPGxpbmUgeDE9IjEzMCIgeTE9IjQ5MiIgeDI9IjEzMCIgeTI9IjUxOCIgc3Ryb2tlPSIjOTk5IiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhaCkiPjwvbGluZT4KCiAgPCEtLSBBZGQgMiAtLT4KICA8Y2lyY2xlIGN4PSIxMzAiIGN5PSI1MzAiIHI9IjEyIiBmaWxsPSIjMjIyIiBzdHJva2U9IiM5OTkiIHN0cm9rZS13aWR0aD0iMS41Ij48L2NpcmNsZT4KICA8dGV4dCB4PSIxMzAiIHk9IjUzNSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2NjYyIgZm9udC1zaXplPSIxNSIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiIGZvbnQtd2VpZ2h0PSJib2xkIj4rPC90ZXh0PgogIDxsaW5lIHgxPSIxMzAiIHkxPSI1NDIiIHgyPSIxMzAiIHkyPSI1NjYiIHN0cm9rZT0iIzk5OSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYWgpIj48L2xpbmU+CgogIDwhLS0gT3V0cHV0IGxhYmVsIC0tPgogIDx0ZXh0IHg9IjEzMCIgeT0iNTg0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjEzIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj5vdXRwdXQgKHNlcV9sZW4sIDc2OCk8L3RleHQ+Cjwvc3ZnPg==)

The block uses GPT-2's **pre-norm** architecture: LayerNorm is applied *before* each sub-layer (attention and feed-forward), not after. At a high level:

$$
\begin{aligned}
x' &= x + \text{MultiHeadAttn}\!\left(\text{LN}_1(x)\right) \\[4pt]
\text{output} &= x' + \text{FeedForward}\!\left(\text{LN}_2(x')\right)
\end{aligned}
$$

where the sub-layers are defined as:

$$
\begin{aligned}
\text{LN}(z) &= \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta, \quad \mu = \frac{1}{d}\sum_i z_i, \quad \sigma^2 = \frac{1}{d}\sum_i (z_i - \mu)^2 \\[8pt]
[Q \mid K \mid V] &= \text{LN}_1(x) \cdot W_{qkv} + b_{qkv} \\[4pt]
\text{head}_i &= \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i, \quad d_k = 64 \\[4pt]
\text{MultiHeadAttn}(z) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_{12}) \cdot W_{\text{attn}} + b_{\text{attn}} \\[8pt]
\text{FeedForward}(z) &= \text{GELU}\!\left(z \cdot W_{fc} + b_{fc}\right) \cdot W_{\text{proj}} + b_{\text{proj}}
\end{aligned}
$$

Expanding into individual steps:

1.  **Layer Norm 1:** $x_{\text{norm}} = \text{LN}_1(x)$ with parameters $\gamma_1, \beta_1$
2.  **QKV Projection:** $QKV = x_{\text{norm}} \cdot W_{qkv} + b_{qkv}$, split into $Q, K, V$ each of shape `(seq_len, 768)`
3.  **Multi-Head Attention:** Reshape $Q, K, V$ into 12 heads of dimension 64, compute per-head scaled dot-product attention (no causal mask), then concatenate heads into $A$
4.  **Output Projection:** $P = A \cdot W_{\text{attn}} + b_{\text{attn}}$
5.  **Residual 1:** $x' = x + P$
6.  **Layer Norm 2:** $h_{\text{norm}} = \text{LN}_2(x')$ with parameters $\gamma_2, \beta_2$
7.  **Feed-Forward:** $F = \text{GELU}(h_{\text{norm}} \cdot W_{fc} + b_{fc}) \cdot W_{\text{proj}} + b_{\text{proj}}$
8.  **Residual 2:** $\text{output} = x' + F$

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` tensor
- LayerNorm uses $\epsilon = 10^{-5}$
- Use the <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html" target="_blank">GELU tanh approximation</a>: $\text{GELU}(x) = 0.5\,x\!\left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\left(x + 0.044715\,x^3\right)\right)\right)$

## Weight Layout

All block parameters are packed into a single contiguous `weights` buffer (7,087,872 floats) in the following order. Index into the buffer using the offsets below (e.g. $W_{qkv}[i][j]$ is at `weights[1536 + i * 2304 + j]`). All 2D matrices are stored in row-major order.

| Parameter                            | Shape       |      Size |    Offset |
|--------------------------------------|-------------|----------:|----------:|
| $\gamma_1$ (LN1 weight) | (768,)      |       768 |         0 |
| $\beta_1$ (LN1 bias)   | (768,)      |       768 |       768 |
| $W_{qkv}$              | (768, 2304) | 1,769,472 |     1,536 |
| $b_{qkv}$              | (2304,)     |     2,304 | 1,771,008 |
| $W_{\text{attn}}$              | (768, 768)  |   589,824 | 1,773,312 |
| $b_{\text{attn}}$              | (768,)      |       768 | 2,363,136 |
| $\gamma_2$ (LN2 weight) | (768,)      |       768 | 2,363,904 |
| $\beta_2$ (LN2 bias)   | (768,)      |       768 | 2,364,672 |
| $W_{fc}$              | (768, 3072) | 2,359,296 | 2,365,440 |
| $b_{fc}$              | (3072,)     |     3,072 | 4,724,736 |
| $W_{\text{proj}}$              | (3072, 768) | 2,359,296 | 4,727,808 |
| $b_{\text{proj}}$              | (768,)      |       768 | 7,087,104 |

## Example

With `seq_len` = 4, `x` uniformly drawn from \[−1, 1\], and weights randomly initialized (see Weight Layout for the packing structure):

    Input:  x.shape       = (4, 768)       # 4 token embeddings
            weights.shape = (7,087,872,)   # packed weight buffer
            seq_len       = 4
    Output: output.shape  = (4, 768)       # transformed token embeddings

## Constraints

- `d_model` = 768, `n_heads` = 12, `ffn_dim` = 3,072 (GPT-2 124M architecture)
- 1 ≤ `seq_len` ≤ 4,096
- All tensors use 32-bit floating point
- Performance is measured with `seq_len` = 1,024
