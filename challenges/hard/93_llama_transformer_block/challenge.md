Implement a single Llama-style transformer decoder block. Given an input tensor $x$ of shape `(seq_len, 512)`, a packed weight buffer, and precomputed RoPE tables, compute the output using pre-norm architecture with Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), and a SwiGLU feed-forward network.

![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdib3g9IjAgMCAzNDAgNjYwIiB3aWR0aD0iMzQwIiBoZWlnaHQ9IjY2MCIgc3R5bGU9ImRpc3BsYXk6YmxvY2s7IG1hcmdpbjoyMHB4IGF1dG87Ij4KICA8ZGVmcz4KICAgIDxtYXJrZXIgaWQ9ImFoIiB2aWV3Ym94PSIwIDAgMTAgMTAiIHJlZng9IjkiIHJlZnk9IjUiIG1hcmtlcndpZHRoPSI2IiBtYXJrZXJoZWlnaHQ9IjYiIG9yaWVudD0iYXV0by1zdGFydC1yZXZlcnNlIj4KICAgICAgPHBhdGggZD0iTTAgMEwxMCA1TDAgMTB6IiBmaWxsPSIjOTk5IiAvPgogICAgPC9tYXJrZXI+CiAgPC9kZWZzPgogIDxyZWN0IHdpZHRoPSIzNDAiIGhlaWdodD0iNjYwIiBmaWxsPSIjMjIyIiAvPgoKICA8IS0tIElucHV0IGxhYmVsIC0tPgogIDx0ZXh0IHg9IjE0MCIgeT0iMjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNjY2MiIGZvbnQtc2l6ZT0iMTMiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiPnggKHNlcV9sZW4sIDUxMik8L3RleHQ+CgogIDwhLS0gQXJyb3c6IGlucHV0IC0+IFJNU05vcm0xIC0tPgogIDxsaW5lIHgxPSIxNDAiIHkxPSIyOCIgeDI9IjE0MCIgeTI9IjQ2IiBzdHJva2U9IiM5OTkiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FoKSI+PC9saW5lPgoKICA8IS0tIFJlc2lkdWFsIDE6IGZvcmsgcmlnaHQsIGRvd24sIGJhY2sgbGVmdCB0byBBZGQxIC0tPgogIDxsaW5lIHgxPSIxNDAiIHkxPSIzNiIgeDI9IjI4MCIgeTI9IjM2IiBzdHJva2U9IiM1NTUiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtZGFzaGFycmF5PSI1LDQiPjwvbGluZT4KICA8bGluZSB4MT0iMjgwIiB5MT0iMzYiIHgyPSIyODAiIHkyPSIzMDYiIHN0cm9rZT0iIzU1NSIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1kYXNoYXJyYXk9IjUsNCI+PC9saW5lPgogIDxsaW5lIHgxPSIyODAiIHkxPSIzMDYiIHgyPSIxNTciIHkyPSIzMDYiIHN0cm9rZT0iIzU1NSIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1kYXNoYXJyYXk9IjUsNCIgbWFya2VyLWVuZD0idXJsKCNhaCkiPjwvbGluZT4KICA8dGV4dCB4PSIyOTAiIHk9IjE3NSIgZmlsbD0iIzY2NiIgZm9udC1zaXplPSIxMCIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiIHRyYW5zZm9ybT0icm90YXRlKDkwLDI5MCwxNzUpIj5yZXNpZHVhbDwvdGV4dD4KCiAgPCEtLSBSTVNOb3JtMSAtLT4KICA8cmVjdCB4PSI2MCIgeT0iNDkiIHdpZHRoPSIxNjAiIGhlaWdodD0iMzAiIHJ4PSI1IiBmaWxsPSIjMzMzIiBzdHJva2U9IiM3NzciIHN0cm9rZS13aWR0aD0iMSIgLz4KICA8dGV4dCB4PSIxNDAiIHk9IjY5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjEyIiBmb250LWZhbWlseT0ic2Fucy1zZXJpZiI+Uk1TTm9ybSAxPC90ZXh0PgogIDxsaW5lIHgxPSIxNDAiIHkxPSI3OSIgeDI9IjE0MCIgeTI9Ijk3IiBzdHJva2U9IiM5OTkiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FoKSI+PC9saW5lPgoKICA8IS0tIFFLViBQcm9qIC0tPgogIDxyZWN0IHg9IjYwIiB5PSIxMDAiIHdpZHRoPSIxNjAiIGhlaWdodD0iMzAiIHJ4PSI1IiBmaWxsPSIjMWUyZDRkIiBzdHJva2U9IiM0NDc3YmIiIHN0cm9rZS13aWR0aD0iMSIgLz4KICA8dGV4dCB4PSIxNDAiIHk9IjEyMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2FhY2NlZSIgZm9udC1zaXplPSIxMiIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiPlFLViBQcm9qZWN0aW9uIChHUUEpPC90ZXh0PgogIDxsaW5lIHgxPSIxNDAiIHkxPSIxMzAiIHgyPSIxNDAiIHkyPSIxNDgiIHN0cm9rZT0iIzk5OSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYWgpIj48L2xpbmU+CgogIDwhLS0gUm9QRSAtLT4KICA8cmVjdCB4PSI2MCIgeT0iMTUxIiB3aWR0aD0iMTYwIiBoZWlnaHQ9IjMwIiByeD0iNSIgZmlsbD0iIzJkMWU0ZCIgc3Ryb2tlPSIjNzc1NWJiIiBzdHJva2Utd2lkdGg9IjEiIC8+CiAgPHRleHQgeD0iMTQwIiB5PSIxNzEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNjY2FhZWUiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIj5Sb1BFIChRIGFuZCBLKTwvdGV4dD4KICA8bGluZSB4MT0iMTQwIiB5MT0iMTgxIiB4Mj0iMTQwIiB5Mj0iMTk5IiBzdHJva2U9IiM5OTkiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FoKSI+PC9saW5lPgoKICA8IS0tIENhdXNhbCBBdHRuIC0tPgogIDxyZWN0IHg9IjYwIiB5PSIyMDIiIHdpZHRoPSIxNjAiIGhlaWdodD0iMzAiIHJ4PSI1IiBmaWxsPSIjMWUyZDRkIiBzdHJva2U9IiM0NDc3YmIiIHN0cm9rZS13aWR0aD0iMSIgLz4KICA8dGV4dCB4PSIxNDAiIHk9IjIyMiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2FhY2NlZSIgZm9udC1zaXplPSIxMiIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiPkNhdXNhbCBBdHRlbnRpb248L3RleHQ+CiAgPGxpbmUgeDE9IjE0MCIgeTE9IjIzMiIgeDI9IjE0MCIgeTI9IjI1MCIgc3Ryb2tlPSIjOTk5IiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhaCkiPjwvbGluZT4KCiAgPCEtLSBPdXRwdXQgUHJvaiAtLT4KICA8cmVjdCB4PSI2MCIgeT0iMjUzIiB3aWR0aD0iMTYwIiBoZWlnaHQ9IjMwIiByeD0iNSIgZmlsbD0iIzFlMmQ0ZCIgc3Ryb2tlPSIjNDQ3N2JiIiBzdHJva2Utd2lkdGg9IjEiIC8+CiAgPHRleHQgeD0iMTQwIiB5PSIyNzMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNhYWNjZWUiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIj5PdXRwdXQgUHJvamVjdGlvbjwvdGV4dD4KICA8bGluZSB4MT0iMTQwIiB5MT0iMjgzIiB4Mj0iMTQwIiB5Mj0iMjk0IiBzdHJva2U9IiM5OTkiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FoKSI+PC9saW5lPgoKICA8IS0tIEFkZCAxIC0tPgogIDxjaXJjbGUgY3g9IjE0MCIgY3k9IjMwNiIgcj0iMTIiIGZpbGw9IiMyMjIiIHN0cm9rZT0iIzk5OSIgc3Ryb2tlLXdpZHRoPSIxLjUiPjwvY2lyY2xlPgogIDx0ZXh0IHg9IjE0MCIgeT0iMzExIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjE1IiBmb250LWZhbWlseT0ic2Fucy1zZXJpZiIgZm9udC13ZWlnaHQ9ImJvbGQiPis8L3RleHQ+CiAgPGxpbmUgeDE9IjE0MCIgeTE9IjMxOCIgeDI9IjE0MCIgeTI9IjM0MiIgc3Ryb2tlPSIjOTk5IiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhaCkiPjwvbGluZT4KCiAgPCEtLSBSZXNpZHVhbCAyIC0tPgogIDxsaW5lIHgxPSIxNDAiIHkxPSIzMzAiIHgyPSIyODAiIHkyPSIzMzAiIHN0cm9rZT0iIzU1NSIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1kYXNoYXJyYXk9IjUsNCI+PC9saW5lPgogIDxsaW5lIHgxPSIyODAiIHkxPSIzMzAiIHgyPSIyODAiIHkyPSI1ODYiIHN0cm9rZT0iIzU1NSIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1kYXNoYXJyYXk9IjUsNCI+PC9saW5lPgogIDxsaW5lIHgxPSIyODAiIHkxPSI1ODYiIHgyPSIxNTciIHkyPSI1ODYiIHN0cm9rZT0iIzU1NSIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1kYXNoYXJyYXk9IjUsNCIgbWFya2VyLWVuZD0idXJsKCNhaCkiPjwvbGluZT4KICA8dGV4dCB4PSIyOTAiIHk9IjQ2MCIgZmlsbD0iIzY2NiIgZm9udC1zaXplPSIxMCIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiIHRyYW5zZm9ybT0icm90YXRlKDkwLDI5MCw0NjApIj5yZXNpZHVhbDwvdGV4dD4KCiAgPCEtLSBSTVNOb3JtMiAtLT4KICA8cmVjdCB4PSI2MCIgeT0iMzQ1IiB3aWR0aD0iMTYwIiBoZWlnaHQ9IjMwIiByeD0iNSIgZmlsbD0iIzMzMyIgc3Ryb2tlPSIjNzc3IiBzdHJva2Utd2lkdGg9IjEiIC8+CiAgPHRleHQgeD0iMTQwIiB5PSIzNjUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNjY2MiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIj5STVNOb3JtIDI8L3RleHQ+CiAgPGxpbmUgeDE9IjE0MCIgeTE9IjM3NSIgeDI9IjE0MCIgeTI9IjM5MyIgc3Ryb2tlPSIjOTk5IiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhaCkiPjwvbGluZT4KCiAgPCEtLSBHYXRlICsgVXAgUHJvaiAtLT4KICA8cmVjdCB4PSI2MCIgeT0iMzk2IiB3aWR0aD0iMTYwIiBoZWlnaHQ9IjMwIiByeD0iNSIgZmlsbD0iIzFlM2QyZCIgc3Ryb2tlPSIjNDRhYTY2IiBzdHJva2Utd2lkdGg9IjEiIC8+CiAgPHRleHQgeD0iMTQwIiB5PSI0MTYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNhYWVlYmIiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIj5HYXRlICZhbXA7IFVwIFByb2ogKDUxMuKGkjE0MDgpPC90ZXh0PgogIDxsaW5lIHgxPSIxNDAiIHkxPSI0MjYiIHgyPSIxNDAiIHkyPSI0NDQiIHN0cm9rZT0iIzk5OSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYWgpIj48L2xpbmU+CgogIDwhLS0gU2lMVSArIG11bHRpcGx5IC0tPgogIDxyZWN0IHg9IjYwIiB5PSI0NDciIHdpZHRoPSIxNjAiIGhlaWdodD0iMzAiIHJ4PSI1IiBmaWxsPSIjMWUzZDJkIiBzdHJva2U9IiM0NGFhNjYiIHN0cm9rZS13aWR0aD0iMSIgLz4KICA8dGV4dCB4PSIxNDAiIHk9IjQ2NyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2FhZWViYiIgZm9udC1zaXplPSIxMiIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiPlNpTFUoZ2F0ZSkg4oqZIHVwPC90ZXh0PgogIDxsaW5lIHgxPSIxNDAiIHkxPSI0NzciIHgyPSIxNDAiIHkyPSI0OTUiIHN0cm9rZT0iIzk5OSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYWgpIj48L2xpbmU+CgogIDwhLS0gRG93biBQcm9qIC0tPgogIDxyZWN0IHg9IjYwIiB5PSI0OTgiIHdpZHRoPSIxNjAiIGhlaWdodD0iMzAiIHJ4PSI1IiBmaWxsPSIjMWUzZDJkIiBzdHJva2U9IiM0NGFhNjYiIHN0cm9rZS13aWR0aD0iMSIgLz4KICA8dGV4dCB4PSIxNDAiIHk9IjUxOCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2FhZWViYiIgZm9udC1zaXplPSIxMiIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiPkRvd24gUHJvaiAoMTQwOOKGkjUxMik8L3RleHQ+CiAgPGxpbmUgeDE9IjE0MCIgeTE9IjUyOCIgeDI9IjE0MCIgeTI9IjU3NCIgc3Ryb2tlPSIjOTk5IiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhaCkiPjwvbGluZT4KCiAgPCEtLSBBZGQgMiAtLT4KICA8Y2lyY2xlIGN4PSIxNDAiIGN5PSI1ODYiIHI9IjEyIiBmaWxsPSIjMjIyIiBzdHJva2U9IiM5OTkiIHN0cm9rZS13aWR0aD0iMS41Ij48L2NpcmNsZT4KICA8dGV4dCB4PSIxNDAiIHk9IjU5MSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2NjYyIgZm9udC1zaXplPSIxNSIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiIGZvbnQtd2VpZ2h0PSJib2xkIj4rPC90ZXh0PgogIDxsaW5lIHgxPSIxNDAiIHkxPSI1OTgiIHgyPSIxNDAiIHkyPSI2MjIiIHN0cm9rZT0iIzk5OSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYWgpIj48L2xpbmU+CgogIDwhLS0gT3V0cHV0IGxhYmVsIC0tPgogIDx0ZXh0IHg9IjE0MCIgeT0iNjQwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjEzIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj5vdXRwdXQgKHNlcV9sZW4sIDUxMik8L3RleHQ+Cjwvc3ZnPg==)

The block follows Llama's **pre-norm** architecture. Unlike GPT-2, it uses **RMSNorm** (no mean subtraction, no additive bias), **Grouped Query Attention** with 8 query heads and 2 key/value heads, **Rotary Position Embeddings** applied to Q and K, and a **SwiGLU** feed-forward network. None of the linear projections have bias terms.

$$
\begin{aligned}
x' &= x + \text{Attn}\!\left(\text{RMSNorm}_1(x),\; \cos,\; \sin\right) \\[4pt]
\text{output} &= x' + \text{FFN}\!\left(\text{RMSNorm}_2(x')\right)
\end{aligned}
$$

The sub-operations in detail:

$$
\begin{aligned}
\text{RMSNorm}(z, w) &= \frac{z}{\sqrt{\frac{1}{d}\sum_i z_i^2 + \varepsilon}} \odot w, \quad \varepsilon = 10^{-5} \\[8pt]
Q &= \text{RMSNorm}_1(x)\, W_Q^\top \in \mathbb{R}^{T \times 512}, \quad \text{reshape to } (T, 8, 64) \\[4pt]
K &= \text{RMSNorm}_1(x)\, W_K^\top \in \mathbb{R}^{T \times 128}, \quad \text{reshape to } (T, 2, 64) \\[4pt]
V &= \text{RMSNorm}_1(x)\, W_V^\top \in \mathbb{R}^{T \times 128}, \quad \text{reshape to } (T, 2, 64) \\[8pt]
\text{RoPE}(q, \cos, \sin) &: \quad [q_1 \mid q_2] \mapsto [q_1 \odot \cos - q_2 \odot \sin \mid q_1 \odot \sin + q_2 \odot \cos] \\[4pt]
&\quad q_1 = q[\ldots, {:}32],\; q_2 = q[\ldots, {32:}] \\[8pt]
\text{GQA} &: \text{repeat } K,V \text{ along head dim } 4\times \text{ to match 8 Q heads} \\[4pt]
\text{head}_i &= \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{64}} + M_{\text{causal}}\right) V_i \\[8pt]
\text{Attn}(x) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_8)\; W_O^\top \\[8pt]
\text{FFN}(z) &= \bigl(\text{SiLU}(z\, W_{\text{gate}}^\top) \odot z\, W_{\text{up}}^\top\bigr)\; W_{\text{down}}^\top
\end{aligned}
$$

where $M_{\text{causal}}$ is the upper-triangular causal mask ($-\infty$ above the diagonal) and $\text{SiLU}(x) = x \cdot \sigma(x)$.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` tensor
- RMSNorm uses $\varepsilon = 10^{-5}$, no additive bias
- Apply causal masking: position $i$ attends only to positions $\le i$
- Repeat K and V heads $4\times$ (GQA groups) before computing attention
- `cos` and `sin` have shape `(seq_len, 32)` — apply them to both Q and K heads independently

## Weight Layout

All parameters are packed into a single contiguous `weights` buffer (2,819,072 floats) in the order below. All 2-D matrices are stored row-major with shape `(out_dim, in_dim)`. There are no bias terms.

| Parameter                                 | Shape       |    Size |    Offset |
|-------------------------------------------|-------------|--------:|----------:|
| $w_1$ (RMSNorm 1 scale) | (512,)      |     512 |         0 |
| $W_Q$                   | (512, 512)  | 262,144 |       512 |
| $W_K$                   | (128, 512)  |  65,536 |   262,656 |
| $W_V$                   | (128, 512)  |  65,536 |   328,192 |
| $W_O$                   | (512, 512)  | 262,144 |   393,728 |
| $w_2$ (RMSNorm 2 scale) | (512,)      |     512 |   655,872 |
| $W_{\text{gate}}$                   | (1408, 512) | 720,896 |   656,384 |
| $W_{\text{up}}$                   | (1408, 512) | 720,896 | 1,377,280 |
| $W_{\text{down}}$                   | (512, 1408) | 720,896 | 2,098,176 |

## Example

With `seq_len` = 4, `x` drawn uniformly from \[−1, 1\], and randomly initialized weights:

    Input:  x.shape       = (4, 512)       # 4 token hidden states
            weights.shape = (2,819,072,)   # packed weight buffer
            cos.shape     = (4, 32)        # precomputed RoPE cosines
            sin.shape     = (4, 32)        # precomputed RoPE sines
            seq_len       = 4
    Output: output.shape  = (4, 512)       # transformed token hidden states

## Constraints

- `d_model` = 512, `n_q_heads` = 8, `n_kv_heads` = 2, `head_dim` = 64, `ffn_hidden` = 1,408
- 1 ≤ `seq_len` ≤ 4,096
- All tensors use 32-bit floating point
- Performance is measured with `seq_len` = 2,048
