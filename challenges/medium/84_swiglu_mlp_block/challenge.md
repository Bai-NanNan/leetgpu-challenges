Implement the SwiGLU MLP block — the feedforward network used in LLaMA, Mistral, Gemma, and most modern large language models. Given an input matrix `x` of shape `[M, d_model]` and three weight matrices `W_gate`, `W_up` (each `[d_model, d_ffn]`), and `W_down` (`[d_ffn, d_model]`), compute: `output = (SiLU(x × W_gate) ⊙ (x × W_up)) × W_down`, where `SiLU(z) = z × sigmoid(z)` and `⊙` denotes element-wise multiplication. All tensors are `float32`.

![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjgwIiBoZWlnaHQ9IjIyMCIgdmlld2JveD0iMCAwIDY4MCAyMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgc3R5bGU9ImRpc3BsYXk6YmxvY2s7IG1hcmdpbjoyMHB4IGF1dG87IGZvbnQtZmFtaWx5Om1vbm9zcGFjZTsiPgogIDxyZWN0IHdpZHRoPSI2ODAiIGhlaWdodD0iMjIwIiBmaWxsPSIjMjIyIiByeD0iOCIgLz4KICA8ZGVmcz4KICAgIDxtYXJrZXIgaWQ9ImFyciIgbWFya2Vyd2lkdGg9IjgiIG1hcmtlcmhlaWdodD0iOCIgcmVmeD0iNiIgcmVmeT0iMyIgb3JpZW50PSJhdXRvIj4KICAgICAgPHBhdGggZD0iTTAsMCBMMCw2IEw4LDMgeiIgZmlsbD0iIzg4OCIgLz4KICAgIDwvbWFya2VyPgogIDwvZGVmcz4KCiAgPCEtLSB4IGJveCAtLT4KICA8cmVjdCB4PSIxNiIgeT0iODIiIHdpZHRoPSI1NiIgaGVpZ2h0PSI0MCIgcng9IjQiIGZpbGw9IiMyYTRhN2YiIHN0cm9rZT0iIzU1ODhjYyIgc3Ryb2tlLXdpZHRoPSIxLjUiIC8+CiAgPHRleHQgeD0iNDQiIHk9IjEwNiIgZmlsbD0iI2NjYyIgZm9udC1zaXplPSIxMiIgdGV4dC1hbmNob3I9Im1pZGRsZSI+eDwvdGV4dD4KICA8dGV4dCB4PSI0NCIgeT0iMTM2IiBmaWxsPSIjNjY2IiBmb250LXNpemU9IjgiIHRleHQtYW5jaG9yPSJtaWRkbGUiPltNLCBkX21vZGVsXTwvdGV4dD4KCiAgPCEtLSBHYXRlIGJyYW5jaCAodG9wKSAtLT4KICA8bGluZSB4MT0iNzIiIHkxPSI5MiIgeDI9IjEwOCIgeTI9IjUyIiBzdHJva2U9IiM4ODgiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiPjwvbGluZT4KICA8cmVjdCB4PSIxMTAiIHk9IjMyIiB3aWR0aD0iOTAiIGhlaWdodD0iNDAiIHJ4PSI0IiBmaWxsPSIjMmE0YTdmIiBzdHJva2U9IiM1NTg4Y2MiIHN0cm9rZS13aWR0aD0iMS41IiAvPgogIDx0ZXh0IHg9IjE1NSIgeT0iNTYiIGZpbGw9IiNjY2MiIGZvbnQtc2l6ZT0iMTAiIHRleHQtYW5jaG9yPSJtaWRkbGUiPnggwrcgV19nYXRlPC90ZXh0PgogIDx0ZXh0IHg9IjE1NSIgeT0iMjIiIGZpbGw9IiM1NTg4Y2MiIGZvbnQtc2l6ZT0iOSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+Z2F0ZSBwcm9qZWN0aW9uPC90ZXh0PgoKICA8IS0tIFVwIGJyYW5jaCAoYm90dG9tKSAtLT4KICA8bGluZSB4MT0iNzIiIHkxPSIxMTIiIHgyPSIxMDgiIHkyPSIxNTIiIHN0cm9rZT0iIzg4OCIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYXJyKSI+PC9saW5lPgogIDxyZWN0IHg9IjExMCIgeT0iMTMyIiB3aWR0aD0iOTAiIGhlaWdodD0iNDAiIHJ4PSI0IiBmaWxsPSIjMmE0YTdmIiBzdHJva2U9IiM1NTg4Y2MiIHN0cm9rZS13aWR0aD0iMS41IiAvPgogIDx0ZXh0IHg9IjE1NSIgeT0iMTU2IiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjEwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj54IMK3IFdfdXA8L3RleHQ+CiAgPHRleHQgeD0iMTU1IiB5PSIxODQiIGZpbGw9IiM1NTg4Y2MiIGZvbnQtc2l6ZT0iOSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+dXAgcHJvamVjdGlvbjwvdGV4dD4KCiAgPCEtLSBTaGFwZSBsYWJlbHMgYWZ0ZXIgcHJvamVjdGlvbnMgLS0+CiAgPHRleHQgeD0iMTU1IiB5PSI4MiIgZmlsbD0iIzY2NiIgZm9udC1zaXplPSI4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5bTSwgZF9mZm5dPC90ZXh0PgogIDx0ZXh0IHg9IjE1NSIgeT0iMTMwIiBmaWxsPSIjNjY2IiBmb250LXNpemU9IjgiIHRleHQtYW5jaG9yPSJtaWRkbGUiPltNLCBkX2Zmbl08L3RleHQ+CgogIDwhLS0gQXJyb3cgZ2F0ZSDihpIgU2lMVSAtLT4KICA8bGluZSB4MT0iMjAwIiB5MT0iNTIiIHgyPSIyMzgiIHkyPSI1MiIgc3Ryb2tlPSIjODg4IiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhcnIpIj48L2xpbmU+CgogIDwhLS0gU2lMVSBib3ggLS0+CiAgPHJlY3QgeD0iMjQwIiB5PSIzMiIgd2lkdGg9IjYwIiBoZWlnaHQ9IjQwIiByeD0iNCIgZmlsbD0iIzFhNWEzYSIgc3Ryb2tlPSIjNDRhYTY2IiBzdHJva2Utd2lkdGg9IjEuNSIgLz4KICA8dGV4dCB4PSIyNzAiIHk9IjU2IiBmaWxsPSIjY2NjIiBmb250LXNpemU9IjExIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5TaUxVPC90ZXh0PgoKICA8IS0tIEFycm93IFNpTFUg4oaSIGVsZW1lbnQtd2lzZSBtdWx0aXBseSAoZ29lcyBkb3duKSAtLT4KICA8bGluZSB4MT0iMzAwIiB5MT0iNTIiIHgyPSIzNzAiIHkyPSI5MCIgc3Ryb2tlPSIjNDRhYTY2IiBzdHJva2Utd2lkdGg9IjEuNSIgbWFya2VyLWVuZD0idXJsKCNhcnIpIj48L2xpbmU+CgogIDwhLS0gQXJyb3cgdXAgYnJhbmNoIOKGkiBlbGVtZW50LXdpc2UgbXVsdGlwbHkgKGdvZXMgdXApIC0tPgogIDxsaW5lIHgxPSIyMDAiIHkxPSIxNTIiIHgyPSIzNzAiIHkyPSIxMTQiIHN0cm9rZT0iIzU1ODhjYyIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYXJyKSI+PC9saW5lPgoKICA8IS0tIEVsZW1lbnQtd2lzZSBtdWx0aXBseSBib3ggLS0+CiAgPHJlY3QgeD0iMzcyIiB5PSI4MiIgd2lkdGg9IjUwIiBoZWlnaHQ9IjQwIiByeD0iNCIgZmlsbD0iIzVhM2ExYSIgc3Ryb2tlPSIjY2M4ODQ0IiBzdHJva2Utd2lkdGg9IjEuNSIgLz4KICA8dGV4dCB4PSIzOTciIHk9IjEwNyIgZmlsbD0iI2NjYyIgZm9udC1zaXplPSIxNiIgdGV4dC1hbmNob3I9Im1pZGRsZSI+4oqZPC90ZXh0PgoKICA8IS0tIEFycm93IOKKmSDihpIgV19kb3duIC0tPgogIDxsaW5lIHgxPSI0MjIiIHkxPSIxMDIiIHgyPSI0NTgiIHkyPSIxMDIiIHN0cm9rZT0iIzg4OCIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYXJyKSI+PC9saW5lPgoKICA8IS0tIFdfZG93biBib3ggLS0+CiAgPHJlY3QgeD0iNDYwIiB5PSI4MiIgd2lkdGg9Ijg2IiBoZWlnaHQ9IjQwIiByeD0iNCIgZmlsbD0iIzJhNGE3ZiIgc3Ryb2tlPSIjNTU4OGNjIiBzdHJva2Utd2lkdGg9IjEuNSIgLz4KICA8dGV4dCB4PSI1MDMiIHk9IjEwNiIgZmlsbD0iI2NjYyIgZm9udC1zaXplPSIxMCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+wrcgV19kb3duPC90ZXh0PgogIDx0ZXh0IHg9IjUwMyIgeT0iNzYiIGZpbGw9IiM2NjYiIGZvbnQtc2l6ZT0iOCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+W00sIGRfZmZuXTwvdGV4dD4KCiAgPCEtLSBBcnJvdyBXX2Rvd24g4oaSIG91dHB1dCAtLT4KICA8bGluZSB4MT0iNTQ2IiB5MT0iMTAyIiB4Mj0iNTc4IiB5Mj0iMTAyIiBzdHJva2U9IiM4ODgiIHN0cm9rZS13aWR0aD0iMS41IiBtYXJrZXItZW5kPSJ1cmwoI2FycikiPjwvbGluZT4KCiAgPCEtLSBPdXRwdXQgYm94IC0tPgogIDxyZWN0IHg9IjU4MCIgeT0iODIiIHdpZHRoPSI4MCIgaGVpZ2h0PSI0MCIgcng9IjQiIGZpbGw9IiMzYTFhM2EiIHN0cm9rZT0iI2NjNDRjYyIgc3Ryb2tlLXdpZHRoPSIxLjUiIC8+CiAgPHRleHQgeD0iNjIwIiB5PSIxMDYiIGZpbGw9IiNjY2MiIGZvbnQtc2l6ZT0iMTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiPm91dHB1dDwvdGV4dD4KICA8dGV4dCB4PSI2MjAiIHk9IjEzNiIgZmlsbD0iIzY2NiIgZm9udC1zaXplPSI4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5bTSwgZF9tb2RlbF08L3RleHQ+CgogIDwhLS0gU2lMVSBmb3JtdWxhIC0tPgogIDx0ZXh0IHg9IjI3MCIgeT0iMTgiIGZpbGw9IiM0NGFhNjYiIGZvbnQtc2l6ZT0iOSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+eiDCtyBzaWdtb2lkKHopPC90ZXh0Pgo8L3N2Zz4=)

## Implementation Requirements

- Implement the `solve` function with the signature unchanged.
- Do not use external libraries beyond the framework provided.
- Write the result into `output` in-place.

## Example

Input: `M` = 2, `d_model` = 2, `d_ffn` = 4

$x$ (float32, $2 \times 2$): 

$$
x = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}
$$

 $W_\text{gate}$ and $W_\text{up}$ (both $2 \times 4$): 

$$
W_\text{gate} = W_\text{up} =
  \begin{bmatrix}
  1.0 & 0.0 & 0.0 & 0.0 \\
  0.0 & 1.0 & 0.0 & 0.0
  \end{bmatrix}
$$

 $W_\text{down}$ ($4 \times 2$): 

$$
W_\text{down} =
  \begin{bmatrix}
  1.0 & 0.0 \\
  0.0 & 1.0 \\
  0.0 & 0.0 \\
  0.0 & 0.0
  \end{bmatrix}
$$

Intermediate steps: 

$$
\text{gate} = x \cdot W_\text{gate} =
  \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 0.0 \end{bmatrix}
$$

 

$$
\text{up} = x \cdot W_\text{up} =
  \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 0.0 \end{bmatrix}
$$

 

$$
\text{SiLU}(1.0) = 1.0 \times \sigma(1.0) \approx 0.7311
$$

 

$$
\text{hidden} = \text{SiLU}(\text{gate}) \odot \text{up} =
  \begin{bmatrix} 0.7311 & 0.0 & 0.0 & 0.0 \\ 0.0 & 0.7311 & 0.0 & 0.0 \end{bmatrix}
$$

Output: 

$$
\text{output} = \text{hidden} \cdot W_\text{down} \approx
  \begin{bmatrix} 0.7311 & 0.0 \\ 0.0 & 0.7311 \end{bmatrix}
$$

## Constraints

- 1 ≤ `M` ≤ 65,536
- 1 ≤ `d_model` ≤ 8,192
- 1 ≤ `d_ffn` ≤ 32,768
- All tensors are `float32` on the GPU.
- Input values are in the range \[-10, 10\].
- Performance is measured with `M` = 512, `d_model` = 4,096, `d_ffn` = 14,336
