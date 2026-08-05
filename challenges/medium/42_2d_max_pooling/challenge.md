Implement a 2D max pooling operation for image/feature map downsampling. The program should take an input tensor and produce an output tensor by applying max pooling with specified kernel size, stride, and padding.

![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdib3g9IjAgMCA0MjAgMTgwIiBzdHlsZT0iZGlzcGxheTpibG9jazsgbWFyZ2luOjIwcHggYXV0bzsiIHdpZHRoPSI0MjAiIGhlaWdodD0iMTgwIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIj4KICA8cmVjdCB3aWR0aD0iNDIwIiBoZWlnaHQ9IjE4MCIgcng9IjgiIGZpbGw9IiMyMjIiIC8+CiAgPGRlZnM+CiAgICA8bWFya2VyIGlkPSJhcnJwb29sIiBtYXJrZXJ3aWR0aD0iOCIgbWFya2VyaGVpZ2h0PSI2IiByZWZ4PSI4IiByZWZ5PSIzIiBvcmllbnQ9ImF1dG8iPgogICAgICA8cG9seWdvbiBwb2ludHM9IjAgMCwgOCAzLCAwIDYiIGZpbGw9IiNjY2MiPjwvcG9seWdvbj4KICAgIDwvbWFya2VyPgogIDwvZGVmcz4KCiAgPCEtLSBJbnB1dCBsYWJlbCAtLT4KICA8dGV4dCB4PSI3NSIgeT0iMTgiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNjY2MiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtd2VpZ2h0PSJib2xkIj5JbnB1dCAoNHg0KTwvdGV4dD4KCiAgPCEtLSBJbnB1dCBncmlkIGJhY2tncm91bmQgLS0+CiAgPHJlY3QgeD0iMTUiIHk9IjI0IiB3aWR0aD0iMTIwIiBoZWlnaHQ9IjEyMCIgcng9IjMiIGZpbGw9IiMzMzMiIHN0cm9rZT0iIzU1NSIgc3Ryb2tlLXdpZHRoPSIwLjUiIC8+CgogIDwhLS0gUmVnaW9uIGhpZ2hsaWdodHMgKDJ4MiBzdHJpZGU9Mikgd2l0aCBkYXNoZWQgYm9yZGVycyAtLT4KICA8IS0tIFRvcC1sZWZ0IHJlZ2lvbiAoZ3JlZW4pIC0tPgogIDxyZWN0IHg9IjE1IiB5PSIyNCIgd2lkdGg9IjYwIiBoZWlnaHQ9IjYwIiBmaWxsPSIjMWUzZDJkIiBvcGFjaXR5PSIwLjUiIC8+CiAgPHJlY3QgeD0iMTUiIHk9IjI0IiB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzQ0YWE2NiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1kYXNoYXJyYXk9IjQsMiIgLz4KICA8IS0tIFRvcC1yaWdodCByZWdpb24gKGJsdWUpIC0tPgogIDxyZWN0IHg9Ijc1IiB5PSIyNCIgd2lkdGg9IjYwIiBoZWlnaHQ9IjYwIiBmaWxsPSIjMWUyZDNkIiBvcGFjaXR5PSIwLjUiIC8+CiAgPHJlY3QgeD0iNzUiIHk9IjI0IiB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzQ0ODhjYyIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1kYXNoYXJyYXk9IjQsMiIgLz4KICA8IS0tIEJvdHRvbS1sZWZ0IHJlZ2lvbiAoYW1iZXIpIC0tPgogIDxyZWN0IHg9IjE1IiB5PSI4NCIgd2lkdGg9IjYwIiBoZWlnaHQ9IjYwIiBmaWxsPSIjM2QyZDFlIiBvcGFjaXR5PSIwLjUiIC8+CiAgPHJlY3QgeD0iMTUiIHk9Ijg0IiB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2NjODg0NCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1kYXNoYXJyYXk9IjQsMiIgLz4KICA8IS0tIEJvdHRvbS1yaWdodCByZWdpb24gKHB1cnBsZSkgLS0+CiAgPHJlY3QgeD0iNzUiIHk9Ijg0IiB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIGZpbGw9IiMzZDFlM2QiIG9wYWNpdHk9IjAuNSIgLz4KICA8cmVjdCB4PSI3NSIgeT0iODQiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYWE0NGFhIiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWRhc2hhcnJheT0iNCwyIiAvPgoKICA8IS0tIEdyaWQgbGluZXMgLS0+CiAgPGxpbmUgeDE9IjQ1IiB5MT0iMjQiIHgyPSI0NSIgeTI9IjE0NCIgc3Ryb2tlPSIjNTU1IiBzdHJva2Utd2lkdGg9IjAuNSI+PC9saW5lPgogIDxsaW5lIHgxPSI3NSIgeTE9IjI0IiB4Mj0iNzUiIHkyPSIxNDQiIHN0cm9rZT0iIzU1NSIgc3Ryb2tlLXdpZHRoPSIwLjUiPjwvbGluZT4KICA8bGluZSB4MT0iMTA1IiB5MT0iMjQiIHgyPSIxMDUiIHkyPSIxNDQiIHN0cm9rZT0iIzU1NSIgc3Ryb2tlLXdpZHRoPSIwLjUiPjwvbGluZT4KICA8bGluZSB4MT0iMTUiIHkxPSI1NCIgeDI9IjEzNSIgeTI9IjU0IiBzdHJva2U9IiM1NTUiIHN0cm9rZS13aWR0aD0iMC41Ij48L2xpbmU+CiAgPGxpbmUgeDE9IjE1IiB5MT0iODQiIHgyPSIxMzUiIHkyPSI4NCIgc3Ryb2tlPSIjNTU1IiBzdHJva2Utd2lkdGg9IjAuNSI+PC9saW5lPgogIDxsaW5lIHgxPSIxNSIgeTE9IjExNCIgeDI9IjEzNSIgeTI9IjExNCIgc3Ryb2tlPSIjNTU1IiBzdHJva2Utd2lkdGg9IjAuNSI+PC9saW5lPgoKICA8IS0tIElucHV0IHZhbHVlcyByb3cgMTogMSwzLDIsNCAtLT4KICA8dGV4dCB4PSIzMCIgeT0iNDMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiM5OTkiIGZvbnQtc2l6ZT0iMTIiPjE8L3RleHQ+CiAgPHRleHQgeD0iNjAiIHk9IjQzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOTk5IiBmb250LXNpemU9IjEyIj4zPC90ZXh0PgogIDx0ZXh0IHg9IjkwIiB5PSI0MyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzk5OSIgZm9udC1zaXplPSIxMiI+MjwvdGV4dD4KICA8dGV4dCB4PSIxMjAiIHk9IjQzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOTk5IiBmb250LXNpemU9IjEyIj40PC90ZXh0PgogIDwhLS0gUm93IDI6IDUsOCw2LDcgLS0+CiAgPHRleHQgeD0iMzAiIHk9IjczIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOTk5IiBmb250LXNpemU9IjEyIj41PC90ZXh0PgogIDx0ZXh0IHg9IjYwIiB5PSI3MyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2ZmZiIgZm9udC1zaXplPSIxMiIgZm9udC13ZWlnaHQ9ImJvbGQiPjg8L3RleHQ+CiAgPHRleHQgeD0iOTAiIHk9IjczIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOTk5IiBmb250LXNpemU9IjEyIj42PC90ZXh0PgogIDx0ZXh0IHg9IjEyMCIgeT0iNzMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNmZmYiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtd2VpZ2h0PSJib2xkIj43PC90ZXh0PgogIDwhLS0gUm93IDM6IDksMiw0LDMgLS0+CiAgPHRleHQgeD0iMzAiIHk9IjEwMyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2ZmZiIgZm9udC1zaXplPSIxMiIgZm9udC13ZWlnaHQ9ImJvbGQiPjk8L3RleHQ+CiAgPHRleHQgeD0iNjAiIHk9IjEwMyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzk5OSIgZm9udC1zaXplPSIxMiI+MjwvdGV4dD4KICA8dGV4dCB4PSI5MCIgeT0iMTAzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOTk5IiBmb250LXNpemU9IjEyIj40PC90ZXh0PgogIDx0ZXh0IHg9IjEyMCIgeT0iMTAzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOTk5IiBmb250LXNpemU9IjEyIj4zPC90ZXh0PgogIDwhLS0gUm93IDQ6IDEsNiw1LDggLS0+CiAgPHRleHQgeD0iMzAiIHk9IjEzMyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzk5OSIgZm9udC1zaXplPSIxMiI+MTwvdGV4dD4KICA8dGV4dCB4PSI2MCIgeT0iMTMzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOTk5IiBmb250LXNpemU9IjEyIj42PC90ZXh0PgogIDx0ZXh0IHg9IjkwIiB5PSIxMzMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiM5OTkiIGZvbnQtc2l6ZT0iMTIiPjU8L3RleHQ+CiAgPHRleHQgeD0iMTIwIiB5PSIxMzMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNmZmYiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtd2VpZ2h0PSJib2xkIj44PC90ZXh0PgoKICA8IS0tIEFycm93IHdpdGggIm1heCIgbGFiZWwgLS0+CiAgPHRleHQgeD0iMTcyIiB5PSI3NiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2NjYyIgZm9udC1zaXplPSIxMCIgZm9udC1zdHlsZT0iaXRhbGljIj5tYXg8L3RleHQ+CiAgPGxpbmUgeDE9IjE1MCIgeTE9Ijg0IiB4Mj0iMTk1IiB5Mj0iODQiIHN0cm9rZT0iI2NjYyIgc3Ryb2tlLXdpZHRoPSIxLjUiIG1hcmtlci1lbmQ9InVybCgjYXJycG9vbCkiPjwvbGluZT4KCiAgPCEtLSBPdXRwdXQgbGFiZWwgLS0+CiAgPHRleHQgeD0iMjQwIiB5PSI1MiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2NjYyIgZm9udC1zaXplPSIxMSIgZm9udC13ZWlnaHQ9ImJvbGQiPk91dHB1dCAoMngyKTwvdGV4dD4KCiAgPCEtLSBPdXRwdXQgZ3JpZCBiYWNrZ3JvdW5kIC0tPgogIDxyZWN0IHg9IjIxMCIgeT0iNjAiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCIgcng9IjMiIGZpbGw9IiMzMzMiIHN0cm9rZT0iIzU1NSIgc3Ryb2tlLXdpZHRoPSIwLjUiIC8+CgogIDwhLS0gT3V0cHV0IGdyaWQgbGluZXMgLS0+CiAgPGxpbmUgeDE9IjI0MCIgeTE9IjYwIiB4Mj0iMjQwIiB5Mj0iMTIwIiBzdHJva2U9IiM1NTUiIHN0cm9rZS13aWR0aD0iMC41Ij48L2xpbmU+CiAgPGxpbmUgeDE9IjIxMCIgeTE9IjkwIiB4Mj0iMjcwIiB5Mj0iOTAiIHN0cm9rZT0iIzU1NSIgc3Ryb2tlLXdpZHRoPSIwLjUiPjwvbGluZT4KCiAgPCEtLSBPdXRwdXQgcmVnaW9uIGNvbG9yIGNvZGluZyAtLT4KICA8cmVjdCB4PSIyMTAiIHk9IjYwIiB3aWR0aD0iMzAiIGhlaWdodD0iMzAiIGZpbGw9IiMxZTNkMmQiIG9wYWNpdHk9IjAuNSIgLz4KICA8cmVjdCB4PSIyNDAiIHk9IjYwIiB3aWR0aD0iMzAiIGhlaWdodD0iMzAiIGZpbGw9IiMxZTJkM2QiIG9wYWNpdHk9IjAuNSIgLz4KICA8cmVjdCB4PSIyMTAiIHk9IjkwIiB3aWR0aD0iMzAiIGhlaWdodD0iMzAiIGZpbGw9IiMzZDJkMWUiIG9wYWNpdHk9IjAuNSIgLz4KICA8cmVjdCB4PSIyNDAiIHk9IjkwIiB3aWR0aD0iMzAiIGhlaWdodD0iMzAiIGZpbGw9IiMzZDFlM2QiIG9wYWNpdHk9IjAuNSIgLz4KCiAgPCEtLSBPdXRwdXQgdmFsdWVzOiBbWzgsN10sWzksOF1dIC0tPgogIDx0ZXh0IHg9IjIyNSIgeT0iODAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNmZmYiIGZvbnQtc2l6ZT0iMTMiIGZvbnQtd2VpZ2h0PSJib2xkIj44PC90ZXh0PgogIDx0ZXh0IHg9IjI1NSIgeT0iODAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNmZmYiIGZvbnQtc2l6ZT0iMTMiIGZvbnQtd2VpZ2h0PSJib2xkIj43PC90ZXh0PgogIDx0ZXh0IHg9IjIyNSIgeT0iMTEwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjZmZmIiBmb250LXNpemU9IjEzIiBmb250LXdlaWdodD0iYm9sZCI+OTwvdGV4dD4KICA8dGV4dCB4PSIyNTUiIHk9IjExMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2ZmZiIgZm9udC1zaXplPSIxMyIgZm9udC13ZWlnaHQ9ImJvbGQiPjg8L3RleHQ+CgogIDwhLS0gTGVnZW5kIC0tPgogIDx0ZXh0IHg9IjMwMCIgeT0iNzAiIGZpbGw9IiNjY2MiIGZvbnQtc2l6ZT0iOSI+a2VybmVsOiAyeDI8L3RleHQ+CiAgPHRleHQgeD0iMzAwIiB5PSI4NCIgZmlsbD0iI2NjYyIgZm9udC1zaXplPSI5Ij5zdHJpZGU6IDI8L3RleHQ+CiAgPHRleHQgeD0iMzAwIiB5PSI5OCIgZmlsbD0iI2NjYyIgZm9udC1zaXplPSI5Ij5wYWRkaW5nOiAwPC90ZXh0PgoKICA8IS0tIEZvb3RlciBub3RlIC0tPgogIDx0ZXh0IHg9Ijc1IiB5PSIxNjQiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiM2NjYiIGZvbnQtc2l6ZT0iOSI+ZGFzaGVkIGJvcmRlcnMgPSBwb29saW5nIHdpbmRvd3M8L3RleHQ+Cjwvc3ZnPg==)

使用指定的 kernel、stride 和 padding 对图像或特征图执行二维最大池化，以完成下采样，并将每个窗口的最大值写入输出。

## Implementation Requirements / 实现要求

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in tensor `output`

## Max Pooling Operation / 最大池化操作

For each output position (n, c, h_out, w_out), compute the maximum value over the corresponding input window:\
`output[n, c, h_out, w_out] = max(input[n, c, h:h+kernel_size, w:w+kernel_size])`\
where h = h_out \* stride and w = w_out \* stride

## Example 1 / 示例 1:

    Input:  input = [[[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]]]]
            kernel_size = 2
            stride = 1
            padding = 0
    Output: output = [[[[5.0, 6.0],
                        [8.0, 9.0]]]]

## Example 2 / 示例 2:

    Input:  input = [[[[1.0, 2.0, 3.0, 4.0, 5.0],
                       [6.0, 7.0, 8.0, 9.0, 10.0],
                       [11.0, 12.0, 13.0, 14.0, 15.0],
                       [16.0, 17.0, 18.0, 19.0, 20.0],
                       [21.0, 22.0, 23.0, 24.0, 25.0]]]]
            kernel_size = 3
            stride = 1
            padding = 1
    Output: output = [[[[7.0, 8.0, 9.0, 10.0, 10.0],
                        [12.0, 13.0, 14.0, 15.0, 15.0],
                        [17.0, 18.0, 19.0, 20.0, 20.0],
                        [22.0, 23.0, 24.0, 25.0, 25.0],
                        [22.0, 23.0, 24.0, 25.0, 25.0]]]]

## Constraints / 约束

- 1 ≤ N ≤ 100 (batch size)
- 1 ≤ C ≤ 512 (channels)
- 1 ≤ H, W ≤ 1024 (height, width)
- 1 ≤ kernel_size ≤ 16
- 1 ≤ stride ≤ 16
- 0 ≤ padding ≤ 16
- Input and output tensors use float32 precision
- Performance is measured with `N` = 4, `kernel_size` = 3, `stride` = 2
