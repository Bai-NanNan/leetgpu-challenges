Given a weighted directed graph of `N` vertices represented as an `N` × `N` distance matrix, compute the shortest path distance between every pair of vertices using the Floyd-Warshall algorithm. The matrix is stored as a flat array in row-major order: `dist[i * N + j]` is the weight of the directed edge from vertex `i` to vertex `j`. A value of `+infinity` means no direct edge exists. The diagonal is always zero. For each intermediate vertex `k` from `0` to `N - 1` (in order), update all pairs:

给定一个包含 `N` 个顶点的带权有向图，该图由 `N` × `N` 距离矩阵表示。使用 Floyd-Warshall 算法计算任意顶点对之间的最短路径距离。矩阵以行优先的一维数组存储：`dist[i * N + j]` 表示从顶点 `i` 到顶点 `j` 的有向边权重。值 `+infinity` 表示不存在直接边。对角线始终为零。对于按顺序从 `0` 到 `N - 1` 遍历的每个中间顶点 `k`，更新所有顶点对：

$$
\text{output}[i][j] = \min\!\bigl(\text{output}[i][j],\;
      \text{output}[i][k] + \text{output}[k][j]\bigr)
    \quad \forall\, i, j
$$

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted) / 只能使用原生功能（不得使用外部库）
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in `output` / 最终结果必须存储在 `output` 中

## Example / 示例

    Input: N = 4
    dist = [
      0,   5, inf,  10,   // row 0: edges from vertex 0
      inf, 0,   3, inf,   // row 1: edges from vertex 1
      inf, inf, 0,   1,   // row 2: edges from vertex 2
      inf, inf, inf, 0    // row 3: edges from vertex 3
    ]

    Output:
    output = [
      0,   5,   8,   9,   // shortest paths from vertex 0
      inf, 0,   3,   4,   // shortest paths from vertex 1
      inf, inf, 0,   1,   // shortest paths from vertex 2
      inf, inf, inf, 0    // shortest paths from vertex 3
    ]

    Explanation:
    - output[0][2] = 8   (path 0 -> 1 -> 2, cost 5 + 3 = 8)
    - output[0][3] = 9   (path 0 -> 1 -> 2 -> 3, cost 5 + 3 + 1 = 9, beats direct 0 -> 3 = 10)
    - output[1][3] = 4   (path 1 -> 2 -> 3, cost 3 + 1 = 4)

    说明：
    - output[0][2] = 8（路径 0 -> 1 -> 2，代价为 5 + 3 = 8）
    - output[0][3] = 9（路径 0 -> 1 -> 2 -> 3，代价为 5 + 3 + 1 = 9，优于直接边 0 -> 3 = 10）
    - output[1][3] = 4（路径 1 -> 2 -> 3，代价为 3 + 1 = 4）

## Constraints / 约束

- 1 ≤ `N` ≤ 4,096 / `N` 的范围为 1 至 4,096
- Edge weights are finite `float32` values or `+infinity` (no edge) / 边权重为有限的 `float32` 值或 `+infinity`（表示没有边）
- The input contains no negative cycles / 输入图中不存在负权环
- The diagonal satisfies `dist[i * N + i] = 0` for all `i` / 对所有 `i`，对角线满足 `dist[i * N + i] = 0`
- `dist` and `output` are flat arrays of `N × N` floats in row-major order / `dist` 和 `output` 均为按行优先存储的 `N × N` 浮点一维数组
- Performance is measured with `N` = 2,048 / 性能测试使用 `N` = 2,048
