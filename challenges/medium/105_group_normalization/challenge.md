Implement Group Normalization for 4D activation tensors, the normalization layer used by Stable Diffusion U-Nets and many ResNet variants. Given an input tensor `X` of shape `(N, C, H, W)`, the channels are split into `G` contiguous groups of `C/G` channels each. For every `(batch, group)` pair, the mean and variance are computed over all `(C/G) × H × W` elements, the activations are normalized, then scaled and shifted by per-channel parameters `gamma` and `beta`.

For each batch index `n` and group index `g`, let $\mathcal{S}_{n,g} = \{(n, c, h, w) : c \in [g \cdot C/G,\, (g+1) \cdot C/G)\}$. Group Normalization computes: 

$$
\begin{align}
  \mu_{n,g} &= \frac{1}{|\mathcal{S}_{n,g}|} \sum_{(n,c,h,w) \in \mathcal{S}_{n,g}} x_{n,c,h,w} \\
  \sigma_{n,g}^2 &= \frac{1}{|\mathcal{S}_{n,g}|} \sum_{(n,c,h,w) \in \mathcal{S}_{n,g}} (x_{n,c,h,w} - \mu_{n,g})^2 \\
  \hat{x}_{n,c,h,w} &= \frac{x_{n,c,h,w} - \mu_{n,g(c)}}{\sqrt{\sigma_{n,g(c)}^2 + \epsilon}} \\
  y_{n,c,h,w} &= \gamma_c \, \hat{x}_{n,c,h,w} + \beta_c
  \end{align}
$$

 where $g(c) = \lfloor c \cdot G / C \rfloor$ maps a channel to its group.

![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDYwIiBoZWlnaHQ9IjIyMCIgdmlld2JveD0iMCAwIDQ2MCAyMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgc3R5bGU9ImRpc3BsYXk6YmxvY2s7IG1hcmdpbjoyMHB4IGF1dG87Ij4KICA8cmVjdCB3aWR0aD0iNDYwIiBoZWlnaHQ9IjIyMCIgcng9IjgiIGZpbGw9IiMyMjIiIC8+CgogIDx0ZXh0IHg9IjIwIiB5PSIyNCIgZmlsbD0iI2NjYyIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtd2VpZ2h0PSJib2xkIj5DaGFubmVscyBzcGxpdCBpbnRvIEcgZ3JvdXBzIChoZXJlIEM9OCwgRz0yKTwvdGV4dD4KCiAgPCEtLSBHcm91cCAwOiBjaGFubmVscyAwLi4zIC0tPgogIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIwLDQwKSI+CiAgICA8cmVjdCB4PSIwIiB5PSIwIiB3aWR0aD0iMTgwIiBoZWlnaHQ9IjYwIiByeD0iNCIgZmlsbD0iIzFhM2E1YSIgc3Ryb2tlPSIjNGE3YWIwIiBzdHJva2Utd2lkdGg9IjIiIC8+CiAgICA8dGV4dCB4PSI5MCIgeT0iLTYiIGZpbGw9IiM5Y2YiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpemU9IjExIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5ncm91cCAwPC90ZXh0PgogICAgPHJlY3QgeD0iNiIgeT0iNiIgd2lkdGg9IjM4IiBoZWlnaHQ9IjQ4IiByeD0iMyIgZmlsbD0iIzJhNWE4YSIgLz4KICAgIDx0ZXh0IHg9IjI1IiB5PSIzNCIgZmlsbD0iI2NkZSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+YzA8L3RleHQ+CiAgICA8cmVjdCB4PSI0OCIgeT0iNiIgd2lkdGg9IjM4IiBoZWlnaHQ9IjQ4IiByeD0iMyIgZmlsbD0iIzJhNWE4YSIgLz4KICAgIDx0ZXh0IHg9IjY3IiB5PSIzNCIgZmlsbD0iI2NkZSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+YzE8L3RleHQ+CiAgICA8cmVjdCB4PSI5MCIgeT0iNiIgd2lkdGg9IjM4IiBoZWlnaHQ9IjQ4IiByeD0iMyIgZmlsbD0iIzJhNWE4YSIgLz4KICAgIDx0ZXh0IHg9IjEwOSIgeT0iMzQiIGZpbGw9IiNjZGUiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiPmMyPC90ZXh0PgogICAgPHJlY3QgeD0iMTMyIiB5PSI2IiB3aWR0aD0iMzgiIGhlaWdodD0iNDgiIHJ4PSIzIiBmaWxsPSIjMmE1YThhIiAvPgogICAgPHRleHQgeD0iMTUxIiB5PSIzNCIgZmlsbD0iI2NkZSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+YzM8L3RleHQ+CiAgICA8dGV4dCB4PSI5MCIgeT0iNzgiIGZpbGw9IiM5Y2YiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpemU9IjExIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj7OvCwgz4PCsiBvdmVyIChDL0cpw5dIw5dXPC90ZXh0PgogIDwvZz4KCiAgPCEtLSBHcm91cCAxOiBjaGFubmVscyA0Li43IC0tPgogIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDI0MCw0MCkiPgogICAgPHJlY3QgeD0iMCIgeT0iMCIgd2lkdGg9IjE4MCIgaGVpZ2h0PSI2MCIgcng9IjQiIGZpbGw9IiM1YTNhMWEiIHN0cm9rZT0iI2IwN2E0YSIgc3Ryb2tlLXdpZHRoPSIyIiAvPgogICAgPHRleHQgeD0iOTAiIHk9Ii02IiBmaWxsPSIjZmM5IiBmb250LWZhbWlseT0ic2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+Z3JvdXAgMTwvdGV4dD4KICAgIDxyZWN0IHg9IjYiIHk9IjYiIHdpZHRoPSIzOCIgaGVpZ2h0PSI0OCIgcng9IjMiIGZpbGw9IiM4YTVhMmEiIC8+CiAgICA8dGV4dCB4PSIyNSIgeT0iMzQiIGZpbGw9IiNlZGMiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiPmM0PC90ZXh0PgogICAgPHJlY3QgeD0iNDgiIHk9IjYiIHdpZHRoPSIzOCIgaGVpZ2h0PSI0OCIgcng9IjMiIGZpbGw9IiM4YTVhMmEiIC8+CiAgICA8dGV4dCB4PSI2NyIgeT0iMzQiIGZpbGw9IiNlZGMiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiPmM1PC90ZXh0PgogICAgPHJlY3QgeD0iOTAiIHk9IjYiIHdpZHRoPSIzOCIgaGVpZ2h0PSI0OCIgcng9IjMiIGZpbGw9IiM4YTVhMmEiIC8+CiAgICA8dGV4dCB4PSIxMDkiIHk9IjM0IiBmaWxsPSIjZWRjIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjExIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5jNjwvdGV4dD4KICAgIDxyZWN0IHg9IjEzMiIgeT0iNiIgd2lkdGg9IjM4IiBoZWlnaHQ9IjQ4IiByeD0iMyIgZmlsbD0iIzhhNWEyYSIgLz4KICAgIDx0ZXh0IHg9IjE1MSIgeT0iMzQiIGZpbGw9IiNlZGMiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiPmM3PC90ZXh0PgogICAgPHRleHQgeD0iOTAiIHk9Ijc4IiBmaWxsPSIjZmM5IiBmb250LWZhbWlseT0ic2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+zrwsIM+DwrIgb3ZlciAoQy9HKcOXSMOXVzwvdGV4dD4KICA8L2c+CgogIDx0ZXh0IHg9IjIzMCIgeT0iMTYwIiBmaWxsPSIjYWFhIiBmb250LWZhbWlseT0ic2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+RWFjaCBncm91cCBpcyBub3JtYWxpemVkIGluZGVwZW5kZW50bHkgcGVyIGJhdGNoIGVsZW1lbnQsPC90ZXh0PgogIDx0ZXh0IHg9IjIzMCIgeT0iMTc4IiBmaWxsPSIjYWFhIiBmb250LWZhbWlseT0ic2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+dGhlbiBzY2FsZWQgYW5kIHNoaWZ0ZWQgYnkgcGVyLWNoYW5uZWwgZ2FtbWEgYW5kIGJldGEuPC90ZXh0PgogIDx0ZXh0IHg9IjIzMCIgeT0iMjAwIiBmaWxsPSIjODg4IiBmb250LWZhbWlseT0ic2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+Rz0xIHJlZHVjZXMgdG8gTGF5ZXIgTm9ybTsgRz1DIHJlZHVjZXMgdG8gSW5zdGFuY2UgTm9ybS48L3RleHQ+Cjwvc3ZnPg==)

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `Y` tensor

## Example 1:

    Input:  N=1, C=4, H=2, W=2, G=2, eps=1e-5
            X[0,0] = [[1, 1], [1, 1]]
            X[0,1] = [[3, 3], [3, 3]]
            X[0,2] = [[2, 2], [2, 2]]
            X[0,3] = [[6, 6], [6, 6]]
            gamma = [1, 1, 1, 1]
            beta  = [0, 0, 0, 0]
    Output: Y[0,0] = [[-1, -1], [-1, -1]]
            Y[0,1] = [[ 1,  1], [ 1,  1]]
            Y[0,2] = [[-1, -1], [-1, -1]]
            Y[0,3] = [[ 1,  1], [ 1,  1]]
    Note:   Group 0 = channels {0, 1}: mean = 2, var = 1, std = 1
            Group 1 = channels {2, 3}: mean = 4, var = 4, std = 2

## Example 2:

    Input:  N=1, C=2, H=1, W=2, G=2, eps=1e-5
            X[0,0] = [[1, 3]]
            X[0,1] = [[2, 6]]
            gamma = [2, 1]
            beta  = [0, 0]
    Output: Y[0,0] = [[-2,  2]]
            Y[0,1] = [[-1,  1]]
    Note:   G=C, so each channel is its own group (Instance Norm).
            Channel 0: mean=2, var=1, std=1
            Channel 1: mean=4, var=4, std=2

## Constraints

- 1 ≤ `N` ≤ 32
- 1 ≤ `C` ≤ 1,024 and `C` is divisible by `G`
- 1 ≤ `G` ≤ `C`
- 1 ≤ `H`, `W` ≤ 128
- `eps` = 1e-5
- -100.0 ≤ input values ≤ 100.0
- 0.1 ≤ `gamma` values ≤ 10.0
- -10.0 ≤ `beta` values ≤ 10.0
- Performance is measured with `N` = 8, `C` = 512, `H` = 64, `W` = 64, `G` = 32
