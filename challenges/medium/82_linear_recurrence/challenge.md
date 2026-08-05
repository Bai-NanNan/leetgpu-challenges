Given two matrices `a` and `x`, each of shape `[B, L]` (batch size × sequence length), compute the linear recurrence `h` of shape `[B, L]` defined by: `h[b, 0] = x[b, 0]` and `h[b, t] = a[b, t] × h[b, t−1] + x[b, t]` for `t ≥ 1`. All values are `float32`. This operation is the core computational primitive of State Space Models (SSMs) such as Mamba, S4, and H3.

![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQwIiBoZWlnaHQ9IjIwMCIgdmlld2JveD0iMCAwIDY0MCAyMDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgc3R5bGU9ImRpc3BsYXk6YmxvY2s7IG1hcmdpbjoyMHB4IGF1dG87Ij4KICA8cmVjdCB3aWR0aD0iNjQwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iIzIyMiIgcng9IjgiIC8+CiAgPCEtLSBUaXRsZSAtLT4KICA8dGV4dCB4PSIzMjAiIHk9IjI0IiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjEzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj5MaW5lYXIgUmVjdXJyZW5jZTogaFt0XSA9IGFbdF0gwrcgaFt0LTFdICsgeFt0XTwvdGV4dD4KICA8IS0tIEJveGVzIGZvciBoIHZhbHVlcyAtLT4KICA8cmVjdCB4PSI0MCIgeT0iODAiIHdpZHRoPSI4MCIgaGVpZ2h0PSI0MCIgcng9IjQiIGZpbGw9IiMxYTNhNWMiIHN0cm9rZT0iIzRhOWVmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIC8+CiAgPHJlY3QgeD0iMTgwIiB5PSI4MCIgd2lkdGg9IjgwIiBoZWlnaHQ9IjQwIiByeD0iNCIgZmlsbD0iIzFhM2E1YyIgc3Ryb2tlPSIjNGE5ZWZmIiBzdHJva2Utd2lkdGg9IjEuNSIgLz4KICA8cmVjdCB4PSIzMjAiIHk9IjgwIiB3aWR0aD0iODAiIGhlaWdodD0iNDAiIHJ4PSI0IiBmaWxsPSIjMWEzYTVjIiBzdHJva2U9IiM0YTllZmYiIHN0cm9rZS13aWR0aD0iMS41IiAvPgogIDxyZWN0IHg9IjQ2MCIgeT0iODAiIHdpZHRoPSI4MCIgaGVpZ2h0PSI0MCIgcng9IjQiIGZpbGw9IiMxYTNhNWMiIHN0cm9rZT0iIzRhOWVmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIC8+CiAgPHRleHQgeD0iODAiIHk9IjEwNSIgZmlsbD0iIzRhOWVmZiIgZm9udC1zaXplPSIxMyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSI+aFswXTwvdGV4dD4KICA8dGV4dCB4PSIyMjAiIHk9IjEwNSIgZmlsbD0iIzRhOWVmZiIgZm9udC1zaXplPSIxMyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSI+aFsxXTwvdGV4dD4KICA8dGV4dCB4PSIzNjAiIHk9IjEwNSIgZmlsbD0iIzRhOWVmZiIgZm9udC1zaXplPSIxMyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSI+aFsyXTwvdGV4dD4KICA8dGV4dCB4PSI1MDAiIHk9IjEwNSIgZmlsbD0iIzRhOWVmZiIgZm9udC1zaXplPSIxMyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSI+aFszXTwvdGV4dD4KICA8IS0tIEFycm93cyBiZXR3ZWVuIGggdmFsdWVzIHdpdGggYVt0XSBsYWJlbHMgLS0+CiAgPGRlZnM+CiAgICA8bWFya2VyIGlkPSJhcnIiIG1hcmtlcndpZHRoPSI4IiBtYXJrZXJoZWlnaHQ9IjgiIHJlZng9IjYiIHJlZnk9IjMiIG9yaWVudD0iYXV0byI+CiAgICAgIDxwYXRoIGQ9Ik0wLDAgTDAsNiBMOCwzIFoiIGZpbGw9IiM3ZWM4YTAiIC8+CiAgICA8L21hcmtlcj4KICA8L2RlZnM+CiAgPGxpbmUgeDE9IjEyMCIgeTE9IjEwMCIgeDI9IjE3NiIgeTI9IjEwMCIgc3Ryb2tlPSIjN2VjOGEwIiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhcnIpIj48L2xpbmU+CiAgPHRleHQgeD0iMTQ4IiB5PSI5NCIgZmlsbD0iIzdlYzhhMCIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSI+w5dhWzFdPC90ZXh0PgogIDxsaW5lIHgxPSIyNjAiIHkxPSIxMDAiIHgyPSIzMTYiIHkyPSIxMDAiIHN0cm9rZT0iIzdlYzhhMCIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYXJyKSI+PC9saW5lPgogIDx0ZXh0IHg9IjI4OCIgeT0iOTQiIGZpbGw9IiM3ZWM4YTAiIGZvbnQtc2l6ZT0iMTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiPsOXYVsyXTwvdGV4dD4KICA8bGluZSB4MT0iNDAwIiB5MT0iMTAwIiB4Mj0iNDU2IiB5Mj0iMTAwIiBzdHJva2U9IiM3ZWM4YTAiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiPjwvbGluZT4KICA8dGV4dCB4PSI0MjgiIHk9Ijk0IiBmaWxsPSIjN2VjOGEwIiBmb250LXNpemU9IjExIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj7Dl2FbM108L3RleHQ+CiAgPCEtLSB4W3RdIGlucHV0cyBmcm9tIGJlbG93IC0tPgogIDxsaW5lIHgxPSI4MCIgeTE9IjE1MiIgeDI9IjgwIiB5Mj0iMTI0IiBzdHJva2U9IiNjY2MiIHN0cm9rZS13aWR0aD0iMS4yIiBtYXJrZXItZW5kPSJ1cmwoI2FycikiPjwvbGluZT4KICA8bGluZSB4MT0iMjIwIiB5MT0iMTUyIiB4Mj0iMjIwIiB5Mj0iMTI0IiBzdHJva2U9IiNjY2MiIHN0cm9rZS13aWR0aD0iMS4yIiBtYXJrZXItZW5kPSJ1cmwoI2FycikiPjwvbGluZT4KICA8bGluZSB4MT0iMzYwIiB5MT0iMTUyIiB4Mj0iMzYwIiB5Mj0iMTI0IiBzdHJva2U9IiNjY2MiIHN0cm9rZS13aWR0aD0iMS4yIiBtYXJrZXItZW5kPSJ1cmwoI2FycikiPjwvbGluZT4KICA8bGluZSB4MT0iNTAwIiB5MT0iMTUyIiB4Mj0iNTAwIiB5Mj0iMTI0IiBzdHJva2U9IiNjY2MiIHN0cm9rZS13aWR0aD0iMS4yIiBtYXJrZXItZW5kPSJ1cmwoI2FycikiPjwvbGluZT4KICA8dGV4dCB4PSI4MCIgeT0iMTcwIiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjEyIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj54WzBdPC90ZXh0PgogIDx0ZXh0IHg9IjIyMCIgeT0iMTcwIiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjEyIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj54WzFdPC90ZXh0PgogIDx0ZXh0IHg9IjM2MCIgeT0iMTcwIiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjEyIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj54WzJdPC90ZXh0PgogIDx0ZXh0IHg9IjUwMCIgeT0iMTcwIiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjEyIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj54WzNdPC90ZXh0PgogIDwhLS0gUGx1cyBzaWducyAtLT4KICA8dGV4dCB4PSI4MCIgeT0iMTQ3IiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjEzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj4rPC90ZXh0PgogIDx0ZXh0IHg9IjIyMCIgeT0iMTQ3IiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjEzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj4rPC90ZXh0PgogIDx0ZXh0IHg9IjM2MCIgeT0iMTQ3IiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjEzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj4rPC90ZXh0PgogIDx0ZXh0IHg9IjUwMCIgeT0iMTQ3IiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjEzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj4rPC90ZXh0PgogIDwhLS0gRWxsaXBzaXMgLS0+CiAgPHRleHQgeD0iNTkwIiB5PSIxMDUiIGZpbGw9IiNjY2MiIGZvbnQtc2l6ZT0iMTgiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiPuKApjwvdGV4dD4KPC9zdmc+)

对每个 batch 独立计算线性递推：首项 `h[b,0] = x[b,0]`，之后使用当前 `a`、前一项 `h` 和当前 `x` 更新。该操作是多种 SSM 的核心计算原语。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The result must be stored in the output tensor `h`

## Examples / 示例

Example 1 — exponential decay (`a = 0.5`, single impulse):

$$
a = \begin{bmatrix} 0.5 & 0.5 & 0.5 & 0.5 \end{bmatrix}, \quad
x = \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \end{bmatrix}
$$

 

$$
h = \begin{bmatrix} 1.0 & 0.5 & 0.25 & 0.125 \end{bmatrix}
$$

Example 2 — prefix sum (`a = 1`, unit inputs):

$$
a = \begin{bmatrix} 1.0 & 1.0 & 1.0 & 1.0 \end{bmatrix}, \quad
x = \begin{bmatrix} 1.0 & 1.0 & 1.0 & 1.0 \end{bmatrix}
$$

 

$$
h = \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 \end{bmatrix}
$$

Full example with `B = 2`, `L = 4`:

$$
a = \begin{bmatrix} 0.5 & 0.5 & 0.5 & 0.5 \\ 1.0 & 1.0 & 1.0 & 1.0 \end{bmatrix}, \quad
x = \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 1.0 & 1.0 & 1.0 & 1.0 \end{bmatrix}
$$

 

$$
h = \begin{bmatrix} 1.0 & 0.5 & 0.25 & 0.125 \\ 1.0 & 2.0 & 3.0 & 4.0 \end{bmatrix}
$$

## Constraints / 约束

- 1 ≤ `B` ≤ 256 (batch size)
- 1 ≤ `L` ≤ 65,536 (sequence length)
- All values in `a` and `x` are `float32`
- Performance is measured with `B` = 64, `L` = 16,384
