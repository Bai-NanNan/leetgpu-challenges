Implement the token verification step of speculative decoding. A draft model proposes $T$ tokens; the target model evaluates them in one forward pass and accepts or rejects each. Given $B$ sequences, produce the verified output tokens. Probability tensors are `float32`; token tensors are `int32`.

Notation for each sequence $b$, at each draft position $i = 0, \ldots, T{-}1$:

- $t_i = \texttt{draft_tokens}[b, i]$ — the token proposed by the draft model
- $p_i(v) = \texttt{draft_probs}[b, i, v]$ — draft model's probability for token $v$
- $q_i(v) = \texttt{target_probs}[b, i, v]$ — target model's probability for token $v$
- $u_i = \texttt{uniform_samples}[b, i]$ — pre-generated $U[0,1)$ sample for position $i$

![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjYwIiBoZWlnaHQ9IjMxMCIgdmlld2JveD0iMCAwIDY2MCAzMTAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgc3R5bGU9ImRpc3BsYXk6YmxvY2s7IG1hcmdpbjoyMHB4IGF1dG87IGZvbnQtZmFtaWx5Om1vbm9zcGFjZTsiPgogIDxyZWN0IHdpZHRoPSI2NjAiIGhlaWdodD0iMzEwIiBmaWxsPSIjMjIyIiByeD0iOCIgLz4KCiAgPCEtLSBDb2x1bW4gaGVhZGVycyAtLT4KICA8dGV4dCB4PSIxMDgiIHk9IjE4IiBmaWxsPSIjNjY2IiBmb250LXNpemU9IjkiIHRleHQtYW5jaG9yPSJtaWRkbGUiPnBvcyAwPC90ZXh0PgogIDx0ZXh0IHg9IjI0OCIgeT0iMTgiIGZpbGw9IiM2NjYiIGZvbnQtc2l6ZT0iOSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+cG9zIDE8L3RleHQ+CiAgPHRleHQgeD0iMzg4IiB5PSIxOCIgZmlsbD0iIzY2NiIgZm9udC1zaXplPSI5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5wb3MgMjwvdGV4dD4KICA8dGV4dCB4PSI1MjgiIHk9IjE4IiBmaWxsPSIjNjY2IiBmb250LXNpemU9IjkiIHRleHQtYW5jaG9yPSJtaWRkbGUiPnBvcyAzPC90ZXh0PgoKICA8IS0tIFJvdyAxOiBEcmFmdCB0b2tlbnMgLS0+CiAgPHRleHQgeD0iMTYiIHk9IjQyIiBmaWxsPSIjODg4IiBmb250LXNpemU9IjEwIj5kcmFmdDwvdGV4dD4KICA8cmVjdCB4PSI1NiIgeT0iMjgiIHdpZHRoPSIxMDQiIGhlaWdodD0iMjQiIHJ4PSI0IiBmaWxsPSIjMWUzYTVmIiBzdHJva2U9IiM0NDc3YmIiIHN0cm9rZS13aWR0aD0iMS41IiAvPgogIDx0ZXh0IHg9IjEwOCIgeT0iNDUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiM4ZWM0ZjAiIGZvbnQtc2l6ZT0iMTEiPnTigoA8L3RleHQ+CiAgPHJlY3QgeD0iMTk2IiB5PSIyOCIgd2lkdGg9IjEwNCIgaGVpZ2h0PSIyNCIgcng9IjQiIGZpbGw9IiMxZTNhNWYiIHN0cm9rZT0iIzQ0NzdiYiIgc3Ryb2tlLXdpZHRoPSIxLjUiIC8+CiAgPHRleHQgeD0iMjQ4IiB5PSI0NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzhlYzRmMCIgZm9udC1zaXplPSIxMSI+dOKCgTwvdGV4dD4KICA8cmVjdCB4PSIzMzYiIHk9IjI4IiB3aWR0aD0iMTA0IiBoZWlnaHQ9IjI0IiByeD0iNCIgZmlsbD0iIzFlM2E1ZiIgc3Ryb2tlPSIjNDQ3N2JiIiBzdHJva2Utd2lkdGg9IjEuNSIgLz4KICA8dGV4dCB4PSIzODgiIHk9IjQ1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOGVjNGYwIiBmb250LXNpemU9IjExIj504oKCPC90ZXh0PgogIDxyZWN0IHg9IjQ3NiIgeT0iMjgiIHdpZHRoPSIxMDQiIGhlaWdodD0iMjQiIHJ4PSI0IiBmaWxsPSIjMWUzYTVmIiBzdHJva2U9IiM0NDc3YmIiIHN0cm9rZS13aWR0aD0iMS41IiAvPgogIDx0ZXh0IHg9IjUyOCIgeT0iNDUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiM4ZWM0ZjAiIGZvbnQtc2l6ZT0iMTEiPnTigoM8L3RleHQ+CgogIDwhLS0gUm93IDI6IFByb2JhYmlsaXRpZXMgLS0+CiAgPHRleHQgeD0iMTYiIHk9Ijc2IiBmaWxsPSIjODg4IiBmb250LXNpemU9IjEwIj5wcm9iczwvdGV4dD4KICA8cmVjdCB4PSI1NiIgeT0iNjIiIHdpZHRoPSIxMDQiIGhlaWdodD0iMzQiIHJ4PSI0IiBmaWxsPSIjMWExYTFhIiBzdHJva2U9IiM2NjYiIHN0cm9rZS13aWR0aD0iMSIgLz4KICA8dGV4dCB4PSIxMDgiIHk9Ijc2IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjYzA2MGUwIiBmb250LXNpemU9IjkiPnAodOKCgCkgPSAwLjYwPC90ZXh0PgogIDx0ZXh0IHg9IjEwOCIgeT0iODkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNlMGEwNDAiIGZvbnQtc2l6ZT0iOSI+cSh04oKAKSA9IDAuNTA8L3RleHQ+CgogIDxyZWN0IHg9IjE5NiIgeT0iNjIiIHdpZHRoPSIxMDQiIGhlaWdodD0iMzQiIHJ4PSI0IiBmaWxsPSIjMWExYTFhIiBzdHJva2U9IiM2NjYiIHN0cm9rZS13aWR0aD0iMSIgLz4KICA8dGV4dCB4PSIyNDgiIHk9Ijc2IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjYzA2MGUwIiBmb250LXNpemU9IjkiPnAodOKCgSkgPSAwLjUwPC90ZXh0PgogIDx0ZXh0IHg9IjI0OCIgeT0iODkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNlMGEwNDAiIGZvbnQtc2l6ZT0iOSI+cSh04oKBKSA9IDAuMjA8L3RleHQ+CgogIDxyZWN0IHg9IjMzNiIgeT0iNjIiIHdpZHRoPSIxMDQiIGhlaWdodD0iMzQiIHJ4PSI0IiBmaWxsPSIjMmEyYTJhIiBzdHJva2U9IiM1NTUiIHN0cm9rZS13aWR0aD0iMSIgLz4KICA8dGV4dCB4PSIzODgiIHk9IjgwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjNTU1IiBmb250LXNpemU9IjkiPm5vdCByZWFjaGVkPC90ZXh0PgoKICA8cmVjdCB4PSI0NzYiIHk9IjYyIiB3aWR0aD0iMTA0IiBoZWlnaHQ9IjM0IiByeD0iNCIgZmlsbD0iIzJhMmEyYSIgc3Ryb2tlPSIjNTU1IiBzdHJva2Utd2lkdGg9IjEiIC8+CiAgPHRleHQgeD0iNTI4IiB5PSI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzU1NSIgZm9udC1zaXplPSI5Ij5ub3QgcmVhY2hlZDwvdGV4dD4KCiAgPCEtLSBSb3cgMzogQWxwaGEgKyBhY2NlcHQvcmVqZWN0IC0tPgogIDx0ZXh0IHg9IjE2IiB5PSIxMjQiIGZpbGw9IiM4ODgiIGZvbnQtc2l6ZT0iMTAiPs6xLCB0ZXN0PC90ZXh0PgogIDxyZWN0IHg9IjU2IiB5PSIxMDgiIHdpZHRoPSIxMDQiIGhlaWdodD0iNDAiIHJ4PSI0IiBmaWxsPSIjMWEzYTFhIiBzdHJva2U9IiM0NGFhNjYiIHN0cm9rZS13aWR0aD0iMS41IiAvPgogIDx0ZXh0IHg9IjEwOCIgeT0iMTI0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjYWFhIiBmb250LXNpemU9IjkiPs6xID0gLjUwLy42MCA9IC44MzwvdGV4dD4KICA8dGV4dCB4PSIxMDgiIHk9IjE0MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzQ0YWE2NiIgZm9udC1zaXplPSI5Ij51PTAuMSAmbHQ7IC44MyDinJM8L3RleHQ+CgogIDxyZWN0IHg9IjE5NiIgeT0iMTA4IiB3aWR0aD0iMTA0IiBoZWlnaHQ9IjQwIiByeD0iNCIgZmlsbD0iIzRhMWExYSIgc3Ryb2tlPSIjZTA2MDYwIiBzdHJva2Utd2lkdGg9IjEuNSIgLz4KICA8dGV4dCB4PSIyNDgiIHk9IjEyNCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2FhYSIgZm9udC1zaXplPSI5Ij7OsSA9IC4yMC8uNTAgPSAuNDA8L3RleHQ+CiAgPHRleHQgeD0iMjQ4IiB5PSIxNDAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNlMDYwNjAiIGZvbnQtc2l6ZT0iOSI+dT0wLjcg4omlIC40MCDinJc8L3RleHQ+CgogIDxyZWN0IHg9IjMzNiIgeT0iMTA4IiB3aWR0aD0iMTA0IiBoZWlnaHQ9IjQwIiByeD0iNCIgZmlsbD0iIzJhMmEyYSIgc3Ryb2tlPSIjNTU1IiBzdHJva2Utd2lkdGg9IjEiIC8+CiAgPHRleHQgeD0iMzg4IiB5PSIxMzIiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiM1NTUiIGZvbnQtc2l6ZT0iOSI+c2tpcHBlZDwvdGV4dD4KCiAgPHJlY3QgeD0iNDc2IiB5PSIxMDgiIHdpZHRoPSIxMDQiIGhlaWdodD0iNDAiIHJ4PSI0IiBmaWxsPSIjMmEyYTJhIiBzdHJva2U9IiM1NTUiIHN0cm9rZS13aWR0aD0iMSIgLz4KICA8dGV4dCB4PSI1MjgiIHk9IjEzMiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzU1NSIgZm9udC1zaXplPSI5Ij5za2lwcGVkPC90ZXh0PgoKICA8IS0tIFJvdyA0OiBSZXNhbXBsZSBib3ggLS0+CiAgPHJlY3QgeD0iNTYiIHk9IjE2NCIgd2lkdGg9IjUyNCIgaGVpZ2h0PSIzOCIgcng9IjUiIGZpbGw9IiMxYTFhMWEiIHN0cm9rZT0iI2UwNjA2MCIgc3Ryb2tlLXdpZHRoPSIxIiAvPgogIDx0ZXh0IHg9IjMxOCIgeT0iMTgwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjZTA2MDYwIiBmb250LXNpemU9IjEwIj5yZWplY3QgYXQgcG9zIDEg4oaSIHN0b3AsIHJlc2FtcGxlIGZyb20gYWRqKHYpID0gbWF4KDAsIHEodikg4oiSIHAodikpPC90ZXh0PgogIDx0ZXh0IHg9IjMxOCIgeT0iMTk0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjYWFhIiBmb250LXNpemU9IjkiPm5vcm1hbGl6ZSBhZGosIGludmVyc2UtQ0RGIHNhbXBsZSB1c2luZyB1W2IsIFRdIOKGkiByZXBsYWNlbWVudCB0b2tlbiB04oKB4oCyPC90ZXh0PgoKICA8IS0tIFJvdyA1OiBPdXRwdXQgdG9rZW5zIC0tPgogIDx0ZXh0IHg9IjE2IiB5PSIyMjQiIGZpbGw9IiM4ODgiIGZvbnQtc2l6ZT0iMTAiPm91dHB1dDwvdGV4dD4KICA8cmVjdCB4PSI1NiIgeT0iMjEyIiB3aWR0aD0iMTA0IiBoZWlnaHQ9IjI0IiByeD0iNCIgZmlsbD0iIzFlM2E1ZiIgc3Ryb2tlPSIjNDQ3N2JiIiBzdHJva2Utd2lkdGg9IjEuNSIgLz4KICA8dGV4dCB4PSIxMDgiIHk9IjIyOSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzhlYzRmMCIgZm9udC1zaXplPSIxMSI+dOKCgDwvdGV4dD4KICA8cmVjdCB4PSIxOTYiIHk9IjIxMiIgd2lkdGg9IjEwNCIgaGVpZ2h0PSIyNCIgcng9IjQiIGZpbGw9IiMzYTIwMTAiIHN0cm9rZT0iI2UwYTA0MCIgc3Ryb2tlLXdpZHRoPSIxLjUiIC8+CiAgPHRleHQgeD0iMjQ4IiB5PSIyMjkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNmMGIwNjAiIGZvbnQtc2l6ZT0iMTEiPnTigoHigLI8L3RleHQ+CiAgPHJlY3QgeD0iMzM2IiB5PSIyMTIiIHdpZHRoPSIxMDQiIGhlaWdodD0iMjQiIHJ4PSI0IiBmaWxsPSIjMmEyYTJhIiBzdHJva2U9IiM1NTUiIHN0cm9rZS13aWR0aD0iMSIgLz4KICA8dGV4dCB4PSIzODgiIHk9IjIyOSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzU1NSIgZm9udC1zaXplPSIxMSI+MDwvdGV4dD4KICA8cmVjdCB4PSI0NzYiIHk9IjIxMiIgd2lkdGg9IjEwNCIgaGVpZ2h0PSIyNCIgcng9IjQiIGZpbGw9IiMyYTJhMmEiIHN0cm9rZT0iIzU1NSIgc3Ryb2tlLXdpZHRoPSIxIiAvPgogIDx0ZXh0IHg9IjUyOCIgeT0iMjI5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjNTU1IiBmb250LXNpemU9IjExIj4wPC90ZXh0PgoKICA8IS0tIExlZ2VuZCAtLT4KICA8dGV4dCB4PSIxNiIgeT0iMjYwIiBmaWxsPSIjYzA2MGUwIiBmb250LXNpemU9IjkiPnAgPSBkcmFmdCBwcm9iPC90ZXh0PgogIDx0ZXh0IHg9IjEzMCIgeT0iMjYwIiBmaWxsPSIjZTBhMDQwIiBmb250LXNpemU9IjkiPnEgPSB0YXJnZXQgcHJvYjwvdGV4dD4KICA8dGV4dCB4PSIyNjAiIHk9IjI2MCIgZmlsbD0iI2FhYSIgZm9udC1zaXplPSI5Ij7OsSA9IG1pbigxLCBxL3ApPC90ZXh0PgogIDx0ZXh0IHg9IjQwMCIgeT0iMjYwIiBmaWxsPSIjNDRhYTY2IiBmb250LXNpemU9IjkiPuKWoCBhY2NlcHRlZDwvdGV4dD4KICA8dGV4dCB4PSI0OTAiIHk9IjI2MCIgZmlsbD0iI2UwYTA0MCIgZm9udC1zaXplPSI5Ij7ilqAgcmVzYW1wbGVkPC90ZXh0PgogIDx0ZXh0IHg9IjU5MCIgeT0iMjYwIiBmaWxsPSIjNTU1IiBmb250LXNpemU9IjkiPuKWoCBwYWQ8L3RleHQ+CgogIDwhLS0gQWxsLWFjY2VwdCBub3RlIC0tPgogIDx0ZXh0IHg9IjMzMCIgeT0iMjkwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjODg4IiBmb250LXNpemU9IjkiPklmIGFsbCBUIHRva2VucyBhY2NlcHRlZDogc2FtcGxlIGJvbnVzIHRva2VuIGZyb20gcSBhdCBsYXN0IHBvc2l0aW9uIHVzaW5nIHVbYiwgVF08L3RleHQ+Cjwvc3ZnPg==)

For each sequence $b$, process positions $i = 0, 1, \ldots, T{-}1$ left-to-right:

1.  Compute acceptance probability: $\displaystyle \alpha_i = \min\!\left(1,\; \frac{q_i(t_i)}{p_i(t_i)}\right)$
2.  If $u_i < \alpha_i$: **accept** $t_i$, continue to position $i{+}1$.
3.  If $u_i \ge \alpha_i$: **reject**, stop. Sample replacement from: 

$$
\text{adj}(v) = \frac{\max(0,\; q_i(v) - p_i(v))}{\sum_{v'} \max(0,\; q_i(v') - p_i(v'))}
$$

 using inverse CDF with $r = \texttt{uniform_samples}[b, T]$. If $\text{adj}$ is all zeros, use uniform $1/V$.
4.  If all $T$ tokens accepted: sample a **bonus token** from $q_{T-1}$ using $\texttt{uniform_samples}[b, T]$.

Write results into `output_tokens[b, :]` (shape $[B, T{+}1]$): accepted/resampled tokens fill positions $0$ through the accepted count (inclusive), remaining positions are zero.

## Implementation Requirements

- Implement `solve(draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens, B, T, V)`.
- Do not change the function signature or use external libraries beyond the standard GPU frameworks.
- Write results into the provided `output_tokens` buffer (shape `[B, T+1]`, `int32`).
- Memory layout is row-major: `draft_probs[b, i, v]` is at offset `b*T*V + i*V + v`.
- Inverse CDF sampling: given distribution $\text{adj}$ (already normalized), find the smallest index $k$ where $\sum_{v=0}^{k} \text{adj}(v) \ge r$, where $r = \texttt{uniform_samples}[b, T]$. Clamp the result to $[0, V-1]$.
- If the adjusted distribution is all zeros (i.e., $q_i \le p_i$ everywhere), fall back to the uniform distribution over $V$ tokens.

## Example

Input: $B = 1,\; T = 3,\; V = 4$

$\text{draft_tokens} = [1, 2, 0]$

Draft probabilities $p_i$ and target probabilities $q_i$ per position: 

$$
p_0 = \begin{bmatrix} 0.10 & 0.60 & 0.20 & 0.10 \end{bmatrix}, \quad
  q_0 = \begin{bmatrix} 0.10 & 0.50 & 0.20 & 0.20 \end{bmatrix}
$$

 

$$
p_1 = \begin{bmatrix} 0.10 & 0.20 & 0.50 & 0.20 \end{bmatrix}, \quad
  q_1 = \begin{bmatrix} 0.30 & 0.20 & 0.20 & 0.30 \end{bmatrix}
$$

 

$$
\text{uniform_samples} = \begin{bmatrix} 0.50 & 0.70 & 0.30 & 0.90 \end{bmatrix}
$$

**Position 0** (draft token = 1): $\alpha_0 = \min\!\left(1,\, \frac{q_0(1)}{p_0(1)}\right) = \min\!\left(1,\, \frac{0.50}{0.60}\right) \approx 0.833$. Since $u_0 = 0.50 < 0.833$, **accept** token 1.

**Position 1** (draft token = 2): $\alpha_1 = \min\!\left(1,\, \frac{q_1(2)}{p_1(2)}\right) = \min\!\left(1,\, \frac{0.20}{0.50}\right) = 0.40$. Since $u_1 = 0.70 \ge 0.40$, **reject**. Resample from adjusted distribution: 

$$
\text{adj}(v) = \max(0,\, q_1(v) - p_1(v)) = [0.20,\, 0,\, 0,\, 0.10]
$$

 

$$
\text{normalized} = \left[\tfrac{2}{3},\, 0,\, 0,\, \tfrac{1}{3}\right], \quad
  \text{CDF} = [0.667,\, 0.667,\, 0.667,\, 1.0]
$$

 With $r = \text{uniform_samples}[0, T] = 0.90$, inverse CDF gives token **3**.

Output: 

$$
\text{output_tokens} = \begin{bmatrix} 1 & 3 & 0 & 0 \end{bmatrix}
$$

## Constraints

- 1 ≤ `B` ≤ 256
- 1 ≤ `T` ≤ 16
- 2 ≤ `V` ≤ 131,072
- `draft_probs[b, i, :]` and `target_probs[b, i, :]` are valid probability distributions (non-negative, sum to 1)
- `draft_probs[b, i, draft_tokens[b, i]]` \> 0 for all `b`, `i`
- `uniform_samples` values are in $[0, 1)$
- All floating-point tensors use `float32`; token tensors use `int32`
- Performance is measured with `B` = 64, `T` = 8, `V` = 32,768
