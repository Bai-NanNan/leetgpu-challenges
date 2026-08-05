Implement a **causal depthwise 1D convolution** over a batched sequence tensor `x` of shape `(B, L, D)`, producing an output of the same shape. In a depthwise convolution, each channel `d` is convolved independently using its own kernel `weight[d, :]` — there is no mixing across channels. The convolution is **causal**: output position `l` may only depend on input positions `0, 1, …, l` (past and present), never future positions. This operation is a key component of state-space models such as Mamba, where it is applied before the selective scan to mix local context within each feature channel.

![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdib3g9IjAgMCA0ODAgMjYwIiB3aWR0aD0iNDgwIiBoZWlnaHQ9IjI2MCIgc3R5bGU9ImRpc3BsYXk6YmxvY2s7IG1hcmdpbjoyMHB4IGF1dG87Ij4KICA8ZGVmcz4KICAgIDxtYXJrZXIgaWQ9ImFoIiB2aWV3Ym94PSIwIDAgMTAgMTAiIHJlZng9IjkiIHJlZnk9IjUiIG1hcmtlcndpZHRoPSI2IiBtYXJrZXJoZWlnaHQ9IjYiIG9yaWVudD0iYXV0by1zdGFydC1yZXZlcnNlIj4KICAgICAgPHBhdGggZD0iTTAgMEwxMCA1TDAgMTB6IiBmaWxsPSIjOTk5IiAvPgogICAgPC9tYXJrZXI+CiAgPC9kZWZzPgoKICA8IS0tIEJhY2tncm91bmQgLS0+CiAgPHJlY3Qgd2lkdGg9IjQ4MCIgaGVpZ2h0PSIyNjAiIGZpbGw9IiMyMjIiIHJ4PSI4IiAvPgoKICA8IS0tIFRpdGxlIC0tPgogIDx0ZXh0IHg9IjI0MCIgeT0iMjIiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNjY2MiIGZvbnQtc2l6ZT0iMTMiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXdlaWdodD0iYm9sZCI+Q2F1c2FsIERlcHRod2lzZSBDb252MWQgKEs9Mywgb25lIGNoYW5uZWwgc2hvd24pPC90ZXh0PgoKICA8IS0tIElucHV0IHJvdyBsYWJlbCAtLT4KICA8dGV4dCB4PSIxNCIgeT0iNjgiIGZpbGw9IiNhYWEiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiPnhbZF08L3RleHQ+CgogIDwhLS0gSW5wdXQgY2VsbHM6IHBvc2l0aW9ucyAwLi41IC0tPgogIDxyZWN0IHg9IjUyIiB5PSI1MiIgd2lkdGg9IjQwIiBoZWlnaHQ9IjI4IiBmaWxsPSIjMmEzYTU1IiBzdHJva2U9IiM0NDc3YmIiIHN0cm9rZS13aWR0aD0iMS4yIiByeD0iMyIgLz4KICA8dGV4dCB4PSI3MiIgeT0iNzEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNhYWNjZWUiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiPnjigoA8L3RleHQ+CgogIDxyZWN0IHg9Ijk2IiB5PSI1MiIgd2lkdGg9IjQwIiBoZWlnaHQ9IjI4IiBmaWxsPSIjMmEzYTU1IiBzdHJva2U9IiM0NDc3YmIiIHN0cm9rZS13aWR0aD0iMS4yIiByeD0iMyIgLz4KICA8dGV4dCB4PSIxMTYiIHk9IjcxIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjYWFjY2VlIiBmb250LXNpemU9IjEyIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj544oKBPC90ZXh0PgoKICA8cmVjdCB4PSIxNDAiIHk9IjUyIiB3aWR0aD0iNDAiIGhlaWdodD0iMjgiIGZpbGw9IiMyYTNhNTUiIHN0cm9rZT0iIzQ0NzdiYiIgc3Ryb2tlLXdpZHRoPSIxLjIiIHJ4PSIzIiAvPgogIDx0ZXh0IHg9IjE2MCIgeT0iNzEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNhYWNjZWUiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiPnjigoI8L3RleHQ+CgogIDxyZWN0IHg9IjE4NCIgeT0iNTIiIHdpZHRoPSI0MCIgaGVpZ2h0PSIyOCIgZmlsbD0iIzJhM2E1NSIgc3Ryb2tlPSIjNDQ3N2JiIiBzdHJva2Utd2lkdGg9IjEuMiIgcng9IjMiIC8+CiAgPHRleHQgeD0iMjA0IiB5PSI3MSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2FhY2NlZSIgZm9udC1zaXplPSIxMiIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSI+eOKCgzwvdGV4dD4KCiAgPHJlY3QgeD0iMjI4IiB5PSI1MiIgd2lkdGg9IjQwIiBoZWlnaHQ9IjI4IiBmaWxsPSIjMmEzYTU1IiBzdHJva2U9IiM0NDc3YmIiIHN0cm9rZS13aWR0aD0iMS4yIiByeD0iMyIgLz4KICA8dGV4dCB4PSIyNDgiIHk9IjcxIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjYWFjY2VlIiBmb250LXNpemU9IjEyIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj544oKEPC90ZXh0PgoKICA8cmVjdCB4PSIyNzIiIHk9IjUyIiB3aWR0aD0iNDAiIGhlaWdodD0iMjgiIGZpbGw9IiMyYTNhNTUiIHN0cm9rZT0iIzQ0NzdiYiIgc3Ryb2tlLXdpZHRoPSIxLjIiIHJ4PSIzIiAvPgogIDx0ZXh0IHg9IjI5MiIgeT0iNzEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNhYWNjZWUiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiPnjigoU8L3RleHQ+CgogIDwhLS0gS2VybmVsIGJveCAtLT4KICA8dGV4dCB4PSIxNCIgeT0iMTM4IiBmaWxsPSIjYWFhIiBmb250LXNpemU9IjExIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj53W2RdPC90ZXh0PgogIDxyZWN0IHg9IjE0MCIgeT0iMTE4IiB3aWR0aD0iNDAiIGhlaWdodD0iMjgiIGZpbGw9IiMxZTNkMmQiIHN0cm9rZT0iIzQ0YWE2NiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHJ4PSIzIiAvPgogIDx0ZXh0IHg9IjE2MCIgeT0iMTM3IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjYWFlZWJiIiBmb250LXNpemU9IjEyIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj534oKAPC90ZXh0PgogIDxyZWN0IHg9IjE4NCIgeT0iMTE4IiB3aWR0aD0iNDAiIGhlaWdodD0iMjgiIGZpbGw9IiMxZTNkMmQiIHN0cm9rZT0iIzQ0YWE2NiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHJ4PSIzIiAvPgogIDx0ZXh0IHg9IjIwNCIgeT0iMTM3IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjYWFlZWJiIiBmb250LXNpemU9IjEyIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj534oKBPC90ZXh0PgogIDxyZWN0IHg9IjIyOCIgeT0iMTE4IiB3aWR0aD0iNDAiIGhlaWdodD0iMjgiIGZpbGw9IiMxZTNkMmQiIHN0cm9rZT0iIzQ0YWE2NiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHJ4PSIzIiAvPgogIDx0ZXh0IHg9IjI0OCIgeT0iMTM3IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjYWFlZWJiIiBmb250LXNpemU9IjEyIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj534oKCPC90ZXh0PgoKICA8IS0tIEFubm90YXRpb246IGtlcm5lbCBhbGlnbmVkIGF0IGw9NCAtLT4KICA8dGV4dCB4PSIxOTAiIHk9IjE1NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzg4OCIgZm9udC1zaXplPSIxMCIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiPmtlcm5lbCBhdCBsPTQ6IHJlYWRzIHjigoIseOKCgyx44oKEPC90ZXh0PgoKICA8IS0tIEFycm93IGZyb20ga2VybmVsIHJlZ2lvbiB0byBvdXRwdXQgLS0+CiAgPGxpbmUgeDE9IjIwNCIgeTE9IjE0NiIgeDI9IjIwNCIgeTI9IjE4MCIgc3Ryb2tlPSIjOTk5IiBzdHJva2Utd2lkdGg9IjEuMiIgbWFya2VyLWVuZD0idXJsKCNhaCkiPjwvbGluZT4KCiAgPCEtLSBPdXRwdXQgcm93IGxhYmVsIC0tPgogIDx0ZXh0IHg9IjE0IiB5PSIyMDgiIGZpbGw9IiNhYWEiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiPnlbZF08L3RleHQ+CgogIDwhLS0gT3V0cHV0IGNlbGxzIC0tPgogIDxyZWN0IHg9IjUyIiB5PSIxOTIiIHdpZHRoPSI0MCIgaGVpZ2h0PSIyOCIgZmlsbD0iIzNhMmEyYSIgc3Ryb2tlPSIjODg0NDQ0IiBzdHJva2Utd2lkdGg9IjEuMiIgcng9IjMiIC8+CiAgPHRleHQgeD0iNzIiIHk9IjIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2VlY2NhYSIgZm9udC1zaXplPSIxMSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSI+eeKCgDwvdGV4dD4KCiAgPHJlY3QgeD0iOTYiIHk9IjE5MiIgd2lkdGg9IjQwIiBoZWlnaHQ9IjI4IiBmaWxsPSIjM2EyYTJhIiBzdHJva2U9IiM4ODQ0NDQiIHN0cm9rZS13aWR0aD0iMS4yIiByeD0iMyIgLz4KICA8dGV4dCB4PSIxMTYiIHk9IjIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2VlY2NhYSIgZm9udC1zaXplPSIxMSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSI+eeKCgTwvdGV4dD4KCiAgPHJlY3QgeD0iMTQwIiB5PSIxOTIiIHdpZHRoPSI0MCIgaGVpZ2h0PSIyOCIgZmlsbD0iIzNhMmEyYSIgc3Ryb2tlPSIjODg0NDQ0IiBzdHJva2Utd2lkdGg9IjEuMiIgcng9IjMiIC8+CiAgPHRleHQgeD0iMTYwIiB5PSIyMTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNlZWNjYWEiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiPnnigoI8L3RleHQ+CgogIDxyZWN0IHg9IjE4NCIgeT0iMTkyIiB3aWR0aD0iNDAiIGhlaWdodD0iMjgiIGZpbGw9IiMzYTJhMmEiIHN0cm9rZT0iI2NjNjY0NCIgc3Ryb2tlLXdpZHRoPSIyIiByeD0iMyIgLz4KICA8dGV4dCB4PSIyMDQiIHk9IjIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2ZmZGRhYSIgZm9udC1zaXplPSIxMSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC13ZWlnaHQ9ImJvbGQiPnnigoM8L3RleHQ+CgogIDxyZWN0IHg9IjIyOCIgeT0iMTkyIiB3aWR0aD0iNDAiIGhlaWdodD0iMjgiIGZpbGw9IiMzYTJhMmEiIHN0cm9rZT0iI2NjNjY0NCIgc3Ryb2tlLXdpZHRoPSIyIiByeD0iMyIgLz4KICA8dGV4dCB4PSIyNDgiIHk9IjIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2ZmZGRhYSIgZm9udC1zaXplPSIxMSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC13ZWlnaHQ9ImJvbGQiPnnigoQ8L3RleHQ+CgogIDxyZWN0IHg9IjI3MiIgeT0iMTkyIiB3aWR0aD0iNDAiIGhlaWdodD0iMjgiIGZpbGw9IiMzYTJhMmEiIHN0cm9rZT0iIzg4NDQ0NCIgc3Ryb2tlLXdpZHRoPSIxLjIiIHJ4PSIzIiAvPgogIDx0ZXh0IHg9IjI5MiIgeT0iMjExIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjZWVjY2FhIiBmb250LXNpemU9IjExIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj554oKFPC90ZXh0PgoKICA8IS0tIEVxdWF0aW9uIGF0IGJvdHRvbSAtLT4KICA8dGV4dCB4PSIyNDAiIHk9IjI0NiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzg4OCIgZm9udC1zaXplPSIxMSIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiPgogICAgeVtkLGxdID0gYmlhc1tkXSArIM6jIHdbZCxrXSDCtyB4W2QsIGziiJJrXSAgICh4W2QsbOKIkmtdID0gMCBpZiBs4oiSayAmbHQ7IDApCiAgPC90ZXh0Pgo8L3N2Zz4=)

Formally, for each batch element `b`, sequence position `l`, and channel `d`:

$$
\text{output}[b,\, l,\, d]
= \text{bias}[d]
+ \sum_{k=0}^{K-1} \text{weight}[d,\, k] \cdot x[b,\, l - k,\, d]
$$

where positions `l − k < 0` are treated as zero (zero-pad the left boundary). The tensor layout is **channels-last**: `x[b, l, d]` is stored at offset `b × L × D + l × D + d`.

这是按通道独立进行的因果一维卷积：位置 `l` 只能使用当前位置及过去的输入，序列左侧越界位置按零处理，输出保持 channels-last 布局。

## Implementation Requirements / 实现要求

- The `solve` function signature must remain unchanged
- The result must be written into the `output` tensor
- Use only native features (external libraries are not permitted)
- Input positions before the start of the sequence (i.e. indices `l − k < 0`) must be treated as zero

## Example / 示例

With `B` = 1, `L` = 4, `D` = 2, `K` = 3:

    x      = [[[1.0, 2.0],    # l=0
               [3.0, 4.0],    # l=1
               [5.0, 6.0],    # l=2
               [7.0, 8.0]]]   # l=3   shape (1, 4, 2)

    weight = [[ 1.0,  0.0, -1.0],   # channel d=0
              [ 1.0,  1.0,  1.0]]   # channel d=1   shape (2, 3)

    bias   = [0.0, 0.0]

    output = [[[1.0,  2.0],   # l=0: d0: 1*1=1          d1: 1*2=2
               [3.0,  6.0],   # l=1: d0: 3*1+1*0=3      d1: 4*1+2*1=6
               [4.0, 12.0],   # l=2: d0: 5*1+3*0+1*(-1)=4  d1: 6+4+2=12
               [4.0, 18.0]]]  # l=3: d0: 7*1+5*0+3*(-1)=4  d1: 8+6+4=18

## Constraints / 约束

- 1 ≤ `B` ≤ 16 (batch size)
- 1 ≤ `L` ≤ 8,192 (sequence length)
- 1 ≤ `D` ≤ 8,192 (number of channels)
- 1 ≤ `K` ≤ 8 (kernel size; typically 3 or 4 in practice)
- All tensors use 32-bit floating point
- Tensor `x` and `output` use channels-last layout: shape `(B, L, D)`
- Performance is measured with `B` = 8, `L` = 2,048, `D` = 4,096, `K` = 4
