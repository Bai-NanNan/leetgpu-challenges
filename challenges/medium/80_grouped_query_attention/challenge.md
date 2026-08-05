Implement Grouped Query Attention (GQA), the attention mechanism used in modern large language models such as LLaMA-3, Mistral, and Gemma. GQA reduces the KV-cache memory footprint during inference by sharing key and value heads across groups of query heads. Given query tensor `Q` with `num_q_heads` heads and key/value tensors `K`, `V` each with `num_kv_heads` heads, compute scaled dot-product attention where every group of `num_q_heads / num_kv_heads` consecutive query heads attends to the same key and value head. All tensors use `float32`.

![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNzAwIiBoZWlnaHQ9IjI2MCIgdmlld2JveD0iMCAwIDcwMCAyNjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgc3R5bGU9ImRpc3BsYXk6YmxvY2s7IG1hcmdpbjoyMHB4IGF1dG87Ij4KICA8cmVjdCB3aWR0aD0iNzAwIiBoZWlnaHQ9IjI2MCIgZmlsbD0iIzIyMiIgcng9IjEwIiAvPgogIDwhLS0gVGl0bGUgLS0+CiAgPHRleHQgeD0iMzUwIiB5PSIyOCIgZmlsbD0iI2NjYyIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxNCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+R3JvdXBlZCBRdWVyeSBBdHRlbnRpb24gKG51bV9xX2hlYWRzPTQsIG51bV9rdl9oZWFkcz0yLCBncm91cHM9Mik8L3RleHQ+CgogIDwhLS0gUSBoZWFkcyAtLT4KICA8dGV4dCB4PSI4MCIgeT0iNjAiIGZpbGw9IiNhYWEiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTIiIHRleHQtYW5jaG9yPSJtaWRkbGUiPlEgaGVhZHM8L3RleHQ+CiAgPHJlY3QgeD0iMjAiIHk9IjcwIiB3aWR0aD0iNjAiIGhlaWdodD0iMzYiIGZpbGw9IiMyNTYzZWIiIHJ4PSI0IiAvPgogIDx0ZXh0IHg9IjUwIiB5PSI5MyIgZmlsbD0iI2ZmZiIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMiIgdGV4dC1hbmNob3I9Im1pZGRsZSI+UVswXTwvdGV4dD4KICA8cmVjdCB4PSIxMDAiIHk9IjcwIiB3aWR0aD0iNjAiIGhlaWdodD0iMzYiIGZpbGw9IiMyNTYzZWIiIHJ4PSI0IiAvPgogIDx0ZXh0IHg9IjEzMCIgeT0iOTMiIGZpbGw9IiNmZmYiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTIiIHRleHQtYW5jaG9yPSJtaWRkbGUiPlFbMV08L3RleHQ+CiAgPHJlY3QgeD0iMTgwIiB5PSI3MCIgd2lkdGg9IjYwIiBoZWlnaHQ9IjM2IiBmaWxsPSIjN2MzYWVkIiByeD0iNCIgLz4KICA8dGV4dCB4PSIyMTAiIHk9IjkzIiBmaWxsPSIjZmZmIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEyIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5RWzJdPC90ZXh0PgogIDxyZWN0IHg9IjI2MCIgeT0iNzAiIHdpZHRoPSI2MCIgaGVpZ2h0PSIzNiIgZmlsbD0iIzdjM2FlZCIgcng9IjQiIC8+CiAgPHRleHQgeD0iMjkwIiB5PSI5MyIgZmlsbD0iI2ZmZiIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMiIgdGV4dC1hbmNob3I9Im1pZGRsZSI+UVszXTwvdGV4dD4KCiAgPCEtLSBLViBoZWFkcyAtLT4KICA8dGV4dCB4PSI4MCIgeT0iMTc1IiBmaWxsPSIjYWFhIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEyIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5LViBoZWFkczwvdGV4dD4KICA8cmVjdCB4PSIyMCIgeT0iMTg1IiB3aWR0aD0iMTIwIiBoZWlnaHQ9IjM2IiBmaWxsPSIjMWQ0ZWQ4IiByeD0iNCIgLz4KICA8dGV4dCB4PSI4MCIgeT0iMjA4IiBmaWxsPSIjZmZmIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEyIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5LWzBdLCBWWzBdPC90ZXh0PgogIDxyZWN0IHg9IjE4MCIgeT0iMTg1IiB3aWR0aD0iMTIwIiBoZWlnaHQ9IjM2IiBmaWxsPSIjNWIyMWI2IiByeD0iNCIgLz4KICA8dGV4dCB4PSIyNDAiIHk9IjIwOCIgZmlsbD0iI2ZmZiIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMiIgdGV4dC1hbmNob3I9Im1pZGRsZSI+S1sxXSwgVlsxXTwvdGV4dD4KCiAgPCEtLSBBcnJvd3MgZ3JvdXAgMCAtLT4KICA8bGluZSB4MT0iNTAiIHkxPSIxMDYiIHgyPSI3MCIgeTI9IjE4NSIgc3Ryb2tlPSIjNjBhNWZhIiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhcnIpIj48L2xpbmU+CiAgPGxpbmUgeDE9IjEzMCIgeTE9IjEwNiIgeDI9IjkwIiB5Mj0iMTg1IiBzdHJva2U9IiM2MGE1ZmEiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiPjwvbGluZT4KCiAgPCEtLSBBcnJvd3MgZ3JvdXAgMSAtLT4KICA8bGluZSB4MT0iMjEwIiB5MT0iMTA2IiB4Mj0iMjMwIiB5Mj0iMTg1IiBzdHJva2U9IiNjNGI1ZmQiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiPjwvbGluZT4KICA8bGluZSB4MT0iMjkwIiB5MT0iMTA2IiB4Mj0iMjUwIiB5Mj0iMTg1IiBzdHJva2U9IiNjNGI1ZmQiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiPjwvbGluZT4KCiAgPCEtLSBPdXRwdXQgYm94ZXMgLS0+CiAgPHRleHQgeD0iODAiIHk9IjI0NSIgZmlsbD0iI2FhYSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+Z3JvdXAgMDwvdGV4dD4KICA8dGV4dCB4PSIyNDAiIHk9IjI0NSIgZmlsbD0iI2FhYSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+Z3JvdXAgMTwvdGV4dD4KCiAgPCEtLSBicmFja2V0IGxhYmVscyAtLT4KICA8dGV4dCB4PSI0MzAiIHk9Ijg4IiBmaWxsPSIjNjBhNWZhIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEyIj5RWzBdLCBRWzFdIGF0dGVuZCB0byBLWzBdLCBWWzBdPC90ZXh0PgogIDx0ZXh0IHg9IjQzMCIgeT0iMTEyIiBmaWxsPSIjYzRiNWZkIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEyIj5RWzJdLCBRWzNdIGF0dGVuZCB0byBLWzFdLCBWWzFdPC90ZXh0PgogIDx0ZXh0IHg9IjQzMCIgeT0iMTUwIiBmaWxsPSIjNGFkZTgwIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEyIj5zY2FsZSA9IDEgLyBzcXJ0KGhlYWRfZGltKTwvdGV4dD4KICA8dGV4dCB4PSI0MzAiIHk9IjE3NCIgZmlsbD0iIzRhZGU4MCIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMiI+c2NvcmVzID0gUSBAIEteVCAqIHNjYWxlPC90ZXh0PgogIDx0ZXh0IHg9IjQzMCIgeT0iMTk4IiBmaWxsPSIjNGFkZTgwIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEyIj53ZWlnaHRzID0gc29mdG1heChzY29yZXMpPC90ZXh0PgogIDx0ZXh0IHg9IjQzMCIgeT0iMjIyIiBmaWxsPSIjNGFkZTgwIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEyIj5vdXRwdXQgPSB3ZWlnaHRzIEAgVjwvdGV4dD4KCiAgPGRlZnM+CiAgICA8bWFya2VyIGlkPSJhcnIiIG1hcmtlcndpZHRoPSI2IiBtYXJrZXJoZWlnaHQ9IjYiIHJlZng9IjMiIHJlZnk9IjMiIG9yaWVudD0iYXV0byI+CiAgICAgIDxwYXRoIGQ9Ik0wLDAgTDAsNiBMNiwzIHoiIGZpbGw9IiM4ODgiIC8+CiAgICA8L21hcmtlcj4KICA8L2RlZnM+Cjwvc3ZnPg==)

实现 GQA：将连续的一组 query heads 映射到同一个 key/value head，计算缩放点积注意力。这样可以在保持多个 query head 的同时减少 KV cache 的内存占用。

## Implementation Requirements / 实现要求

- Implement the function `solve(Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim)`.
- Do not change the function signature or use external libraries beyond the standard GPU frameworks.
- Write the result into the provided `output` buffer.
- `num_q_heads` is always divisible by `num_kv_heads`.
- Use scaled dot-product attention with scale factor `1 / sqrt(head_dim)` and a softmax over the key dimension.

## Example / 示例

With `num_q_heads` = 4, `num_kv_heads` = 2 (groups of 2), `seq_len` = 3, `head_dim` = 4:

**Input:**\
$Q_0$ (3×4): 

$$
\begin{bmatrix}
  1 & 0 & 0 & 1 \\
  0 & 1 & 1 & 0 \\
  1 & 1 & 0 & 0
  \end{bmatrix}
$$

 $Q_1$ (3×4): 

$$
\begin{bmatrix}
  0 & 1 & 0 & 1 \\
  1 & 0 & 1 & 0 \\
  0 & 0 & 1 & 1
  \end{bmatrix}
$$

 $Q_2$ (3×4): 

$$
\begin{bmatrix}
  -1 & 0 & 0.5 & 0 \\
  0 & -1 & 0 & 0.5 \\
  0.5 & 0 & -1 & 0
  \end{bmatrix}
$$

 $Q_3$ (3×4): 

$$
\begin{bmatrix}
  0 & 0.5 & 0 & -1 \\
  0.5 & 0 & 0 & -1 \\
  0 & 0 & 0.5 & 0.5
  \end{bmatrix}
$$

 $K_0$ (3×4): 

$$
\begin{bmatrix}
  1 & 0 & 1 & 0 \\
  0 & 1 & 0 & 1 \\
  1 & 1 & 1 & 1
  \end{bmatrix}
$$

 $K_1$ (3×4): 

$$
\begin{bmatrix}
  0 & 1 & 0 & -1 \\
  -1 & 0 & 1 & 0 \\
  0 & -1 & 0 & 1
  \end{bmatrix}
$$

 $V_0$ (3×4): 

$$
\begin{bmatrix}
  1 & 2 & 3 & 4 \\
  5 & 6 & 7 & 8 \\
  9 & 10 & 11 & 12
  \end{bmatrix}
$$

 $V_1$ (3×4): 

$$
\begin{bmatrix}
  -1 & -2 & -3 & -4 \\
  2 & 3 & 4 & 5 \\
  6 & 7 & 8 & 9
  \end{bmatrix}
$$

 Groups: $Q_0, Q_1 \to K_0, V_0$; \quad $Q_2, Q_3 \to K_1, V_1$

**Output** (values rounded to 2 decimal places):\
$\text{output}_0$ (3×4): 

$$
\begin{bmatrix}
  5.71 & 6.71 & 7.71 & 8.71 \\
  5.71 & 6.71 & 7.71 & 8.71 \\
  5.71 & 6.71 & 7.71 & 8.71
  \end{bmatrix}
$$

 $\text{output}_1$ (3×4): 

$$
\begin{bmatrix}
  6.07 & 7.07 & 8.07 & 9.07 \\
  5.00 & 6.00 & 7.00 & 8.00 \\
  5.71 & 6.71 & 7.71 & 8.71
  \end{bmatrix}
$$

 $\text{output}_2$ (3×4): 

$$
\begin{bmatrix}
  2.24 & 2.76 & 3.27 & 3.79 \\
  3.96 & 4.70 & 5.44 & 6.17 \\
  2.40 & 2.60 & 2.79 & 2.98
  \end{bmatrix}
$$

 $\text{output}_3$ (3×4): 

$$
\begin{bmatrix}
  0.76 & 0.58 & 0.40 & 0.22 \\
  1.17 & 1.08 & 1.00 & 0.91 \\
  2.84 & 3.37 & 3.91 & 4.44
  \end{bmatrix}
$$

## Constraints / 约束

- 1 ≤ `num_kv_heads` ≤ `num_q_heads` ≤ 64
- `num_q_heads` is divisible by `num_kv_heads`
- 1 ≤ `seq_len` ≤ 4,096
- 8 ≤ `head_dim` ≤ 256; `head_dim` is a multiple of 8
- All tensor values are `float32`
- Performance is measured with `num_q_heads` = 32, `num_kv_heads` = 8, `seq_len` = 1,024, `head_dim` = 128
