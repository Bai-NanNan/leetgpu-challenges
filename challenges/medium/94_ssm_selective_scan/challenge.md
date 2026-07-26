Implement the forward pass of a State Space Model (SSM) selective scan, the core operation in Mamba-style sequence models. Given an input sequence `u`, time-step parameters `delta`, state-transition matrix `A`, input projection `B`, output projection `C`, and skip-connection weights `skip`, compute the output sequence `y` in float32.

![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNzAwIiBoZWlnaHQ9IjE4MCIgdmlld2JveD0iMCAwIDcwMCAxODAiIHN0eWxlPSJkaXNwbGF5OmJsb2NrOyBtYXJnaW46MjBweCBhdXRvOyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8cmVjdCB3aWR0aD0iNzAwIiBoZWlnaHQ9IjE4MCIgZmlsbD0iIzIyMiIgcng9IjEwIiAvPgogIDwhLS0gU1NNIGNoYWluIGRpYWdyYW0gLS0+CiAgPCEtLSBTdGF0ZSBib3hlcyAtLT4KICA8cmVjdCB4PSI1NSIgeT0iNzAiIHdpZHRoPSI2MCIgaGVpZ2h0PSI0MCIgcng9IjYiIGZpbGw9IiMxYTNhNWMiIHN0cm9rZT0iIzRhOTBkOSIgc3Ryb2tlLXdpZHRoPSIxLjUiIC8+CiAgPHRleHQgeD0iODUiIHk9Ijk1IiBmaWxsPSIjNGE5MGQ5IiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5o4oKAPC90ZXh0PgoKICA8cmVjdCB4PSIxOTUiIHk9IjcwIiB3aWR0aD0iNjAiIGhlaWdodD0iNDAiIHJ4PSI2IiBmaWxsPSIjMWEzYTVjIiBzdHJva2U9IiM0YTkwZDkiIHN0cm9rZS13aWR0aD0iMS41IiAvPgogIDx0ZXh0IHg9IjIyNSIgeT0iOTUiIGZpbGw9IiM0YTkwZDkiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTMiIHRleHQtYW5jaG9yPSJtaWRkbGUiPmjigoE8L3RleHQ+CgogIDxyZWN0IHg9IjMzNSIgeT0iNzAiIHdpZHRoPSI2MCIgaGVpZ2h0PSI0MCIgcng9IjYiIGZpbGw9IiMxYTNhNWMiIHN0cm9rZT0iIzRhOTBkOSIgc3Ryb2tlLXdpZHRoPSIxLjUiIC8+CiAgPHRleHQgeD0iMzY1IiB5PSI5NSIgZmlsbD0iIzRhOTBkOSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMyIgdGV4dC1hbmNob3I9Im1pZGRsZSI+aOKCgjwvdGV4dD4KCiAgPHJlY3QgeD0iNDc1IiB5PSI3MCIgd2lkdGg9IjYwIiBoZWlnaHQ9IjQwIiByeD0iNiIgZmlsbD0iIzFhM2E1YyIgc3Ryb2tlPSIjNGE5MGQ5IiBzdHJva2Utd2lkdGg9IjEuNSIgLz4KICA8dGV4dCB4PSI1MDUiIHk9Ijk1IiBmaWxsPSIjNGE5MGQ5IiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5o4oKDPC90ZXh0PgoKICA8IS0tIFJlY3VycmVuY2UgYXJyb3dzIC0tPgogIDxsaW5lIHgxPSIxMTUiIHkxPSI5MCIgeDI9IjE5MyIgeTI9IjkwIiBzdHJva2U9IiM0YTkwZDkiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiPjwvbGluZT4KICA8bGluZSB4MT0iMjU1IiB5MT0iOTAiIHgyPSIzMzMiIHkyPSI5MCIgc3Ryb2tlPSIjNGE5MGQ5IiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhcnIpIj48L2xpbmU+CiAgPGxpbmUgeDE9IjM5NSIgeTE9IjkwIiB4Mj0iNDczIiB5Mj0iOTAiIHN0cm9rZT0iIzRhOTBkOSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYXJyKSI+PC9saW5lPgogIDx0ZXh0IHg9IjE1MyIgeT0iODMiIGZpbGw9IiNjY2MiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTAiIHRleHQtYW5jaG9yPSJtaWRkbGUiPsSAPC90ZXh0PgogIDx0ZXh0IHg9IjI5MyIgeT0iODMiIGZpbGw9IiNjY2MiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTAiIHRleHQtYW5jaG9yPSJtaWRkbGUiPsSAPC90ZXh0PgogIDx0ZXh0IHg9IjQzMyIgeT0iODMiIGZpbGw9IiNjY2MiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTAiIHRleHQtYW5jaG9yPSJtaWRkbGUiPsSAPC90ZXh0PgoKICA8IS0tIElucHV0IGFycm93cyAodSBpbnRvIGgpIC0tPgogIDxsaW5lIHgxPSI4NSIgeTE9IjE1NSIgeDI9Ijg1IiB5Mj0iMTEyIiBzdHJva2U9IiM1Y2I4NWMiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2dhcnIpIj48L2xpbmU+CiAgPGxpbmUgeDE9IjIyNSIgeTE9IjE1NSIgeDI9IjIyNSIgeTI9IjExMiIgc3Ryb2tlPSIjNWNiODVjIiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNnYXJyKSI+PC9saW5lPgogIDxsaW5lIHgxPSIzNjUiIHkxPSIxNTUiIHgyPSIzNjUiIHkyPSIxMTIiIHN0cm9rZT0iIzVjYjg1YyIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjZ2FycikiPjwvbGluZT4KICA8bGluZSB4MT0iNTA1IiB5MT0iMTU1IiB4Mj0iNTA1IiB5Mj0iMTEyIiBzdHJva2U9IiM1Y2I4NWMiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2dhcnIpIj48L2xpbmU+CiAgPHRleHQgeD0iODUiIHk9IjE2OCIgZmlsbD0iIzVjYjg1YyIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+QsyEdeKCgDwvdGV4dD4KICA8dGV4dCB4PSIyMjUiIHk9IjE2OCIgZmlsbD0iIzVjYjg1YyIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+QsyEdeKCgTwvdGV4dD4KICA8dGV4dCB4PSIzNjUiIHk9IjE2OCIgZmlsbD0iIzVjYjg1YyIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+QsyEdeKCgjwvdGV4dD4KICA8dGV4dCB4PSI1MDUiIHk9IjE2OCIgZmlsbD0iIzVjYjg1YyIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+QsyEdeKCgzwvdGV4dD4KCiAgPCEtLSBPdXRwdXQgYXJyb3dzIChoIHRvIHkpIC0tPgogIDxsaW5lIHgxPSI4NSIgeTE9IjY4IiB4Mj0iODUiIHkyPSIzMCIgc3Ryb2tlPSIjZTg3YzJlIiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNvYXJyKSI+PC9saW5lPgogIDxsaW5lIHgxPSIyMjUiIHkxPSI2OCIgeDI9IjIyNSIgeTI9IjMwIiBzdHJva2U9IiNlODdjMmUiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI29hcnIpIj48L2xpbmU+CiAgPGxpbmUgeDE9IjM2NSIgeTE9IjY4IiB4Mj0iMzY1IiB5Mj0iMzAiIHN0cm9rZT0iI2U4N2MyZSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjb2FycikiPjwvbGluZT4KICA8bGluZSB4MT0iNTA1IiB5MT0iNjgiIHgyPSI1MDUiIHkyPSIzMCIgc3Ryb2tlPSIjZTg3YzJlIiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNvYXJyKSI+PC9saW5lPgogIDx0ZXh0IHg9Ijg1IiB5PSIyMiIgZmlsbD0iI2U4N2MyZSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+eeKCgDwvdGV4dD4KICA8dGV4dCB4PSIyMjUiIHk9IjIyIiBmaWxsPSIjZTg3YzJlIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjExIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj554oKBPC90ZXh0PgogIDx0ZXh0IHg9IjM2NSIgeT0iMjIiIGZpbGw9IiNlODdjMmUiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiPnnigoI8L3RleHQ+CiAgPHRleHQgeD0iNTA1IiB5PSIyMiIgZmlsbD0iI2U4N2MyZSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+eeKCgzwvdGV4dD4KCiAgPCEtLSBDb250aW51YXRpb24gYXJyb3cgLS0+CiAgPGxpbmUgeDE9IjUzNSIgeTE9IjkwIiB4Mj0iNTkwIiB5Mj0iOTAiIHN0cm9rZT0iIzRhOTBkOSIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1kYXNoYXJyYXk9IjQsMyIgbWFya2VyLWVuZD0idXJsKCNhcnIpIj48L2xpbmU+CgogIDwhLS0gQXJyb3cgbWFya2VycyAtLT4KICA8ZGVmcz4KICAgIDxtYXJrZXIgaWQ9ImFyciIgbWFya2Vyd2lkdGg9IjgiIG1hcmtlcmhlaWdodD0iNiIgcmVmeD0iOCIgcmVmeT0iMyIgb3JpZW50PSJhdXRvIj4KICAgICAgPHBvbHlnb24gcG9pbnRzPSIwIDAsIDggMywgMCA2IiBmaWxsPSIjNGE5MGQ5Ij48L3BvbHlnb24+CiAgICA8L21hcmtlcj4KICAgIDxtYXJrZXIgaWQ9ImdhcnIiIG1hcmtlcndpZHRoPSI4IiBtYXJrZXJoZWlnaHQ9IjYiIHJlZng9IjgiIHJlZnk9IjMiIG9yaWVudD0iYXV0byI+CiAgICAgIDxwb2x5Z29uIHBvaW50cz0iMCAwLCA4IDMsIDAgNiIgZmlsbD0iIzVjYjg1YyI+PC9wb2x5Z29uPgogICAgPC9tYXJrZXI+CiAgICA8bWFya2VyIGlkPSJvYXJyIiBtYXJrZXJ3aWR0aD0iOCIgbWFya2VyaGVpZ2h0PSI2IiByZWZ4PSI4IiByZWZ5PSIzIiBvcmllbnQ9ImF1dG8iPgogICAgICA8cG9seWdvbiBwb2ludHM9IjAgMCwgOCAzLCAwIDYiIGZpbGw9IiNlODdjMmUiPjwvcG9seWdvbj4KICAgIDwvbWFya2VyPgogIDwvZGVmcz4KPC9zdmc+)

## Implementation Requirements

Implement the function `solve(u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state)` with the signature unchanged. Do not use external libraries beyond the allowed framework. Write the result into the pre-allocated output tensor `y`.

For each batch `b`, position `t`, and channel `d`, the computation is:

$$
\bar{A}_{b,t,d,n} = \exp(\Delta_{b,t,d} \cdot A_{d,n})
$$

 

$$
\bar{B}_{b,t,d,n} = \Delta_{b,t,d} \cdot B_{b,t,n}
$$

 

$$
h_{b,t,d,n} = \bar{A}_{b,t,d,n} \cdot h_{b,t-1,d,n} + \bar{B}_{b,t,d,n} \cdot u_{b,t,d}
$$

 

$$
y_{b,t,d} = \sum_{n} C_{b,t,n} \cdot h_{b,t,d,n} + \text{skip}_d \cdot u_{b,t,d}
$$

The initial hidden state $h_{b,-1,d,n} = 0$ for all $b, d, n$. All channels `d` are independent: they share the same `B` and `C` projections but have separate state-transition rows in `A`.

## Example

    Input:
      u     = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]  shape (1,4,2)
      delta = [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]  shape (1,4,2)
      A     = [[-0.5, -1.0], [-0.5, -1.0]]                         shape (2,2)
      B     = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]]  shape (1,4,2)
      C     = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]]  shape (1,4,2)
      skip  = [0.0, 0.0]                                            shape (2,)
      batch=1, seq_len=4, d_model=2, d_state=2

    Derivation (delta=1 everywhere, so A_bar_dn = exp(A_dn)):
      A_bar[d=0] = [exp(-0.5), exp(-1.0)] ≈ [0.607, 0.368]
      A_bar[d=1] = [exp(-0.5), exp(-1.0)] ≈ [0.607, 0.368]

      Hidden state h has shape (d_model=2, d_state=2); initial h = zeros.
      t=0: h = [[1.000, 0.000], [0.000, 0.000]]  →  y[0,0] = [1.000, 0.000]
      t=1: h = [[0.607, 0.000], [0.000, 1.000]]  →  y[0,1] = [0.000, 1.000]
      t=2: h = [[1.368, 1.000], [1.000, 1.368]]  →  y[0,2] = [2.368, 2.368]
      t=3: h = [[0.830, 0.368], [0.607, 0.503]]  →  y[0,3] = [0.599, 0.555]

    Output:
      y = [[[1.000, 0.000], [0.000, 1.000], [2.368, 2.368], [0.599, 0.555]]]

## Constraints

- 1 ≤ `batch` ≤ 16
- 1 ≤ `seq_len` ≤ 8,192
- 1 ≤ `d_model` ≤ 2,048
- 1 ≤ `d_state` ≤ 64
- All entries of `delta` are positive
- All entries of `A` are negative (ensuring `A_bar ∈ (0, 1)`)
- All tensors are float32 on the GPU
- Performance is measured with `batch` = 4, `seq_len` = 4,096, `d_model` = 512, `d_state` = 16
