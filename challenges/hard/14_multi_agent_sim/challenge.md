Implement a program for a multi-agent flocking simulation (boids). The input consists of:

编写一个程序，实现多智能体群聚（Boids）模拟。输入包括：

An array `agents` containing `N` agents, where `N` is the total number of agents

包含 `N` 个智能体的数组 `agents`，其中 `N` 为智能体总数。

Each agent occupies 4 consecutive 32-bit floating point numbers in the array: $[x, y, v_x, v_y]$, where:

数组中每个智能体占用连续的 4 个 32 位浮点数：$[x, y, v_x, v_y]$，其中：

- $(x, y)$ represents the agent's position in 2D space / $(x, y)$ 表示智能体在二维空间中的位置
- $(v_x, v_y)$ represents the agent's velocity vector / $(v_x, v_y)$ 表示智能体的速度向量

The total array size is `4 * N` floats, with agent $i$'s data stored at indices `[4i, 4i+1, 4i+2, 4i+3]`

数组总大小为 `4 * N` 个浮点数，第 $i$ 个智能体的数据存储在索引 `[4i, 4i+1, 4i+2, 4i+3]` 处。

## Simulation Rules / 模拟规则

1.  For each agent $i$, identify all neighbors $j$ (where $i \neq j$) within radius $r = 5.0$ using: 

中文说明：对每个智能体 $i$，使用下式找出半径 $r = 5.0$ 内的所有邻居 $j$（其中 $i \neq j$）：

$$
\sqrt{(x_i - x_j)^2 + (y_i - y_j)^2} < r
$$

2.  Compute average velocity of neighboring agents: 

中文说明：计算邻居智能体的平均速度：

$$
\vec{v}_{avg} = \begin{cases}
      \frac{1}{|N_i|} \sum_{j \in N_i} \vec{v}_j & \text{if } |N_i| > 0 \\
      \vec{v}_i & \text{if } |N_i| = 0
      \end{cases}
$$

 where $N_i$ is the set of neighbors for agent $i$

其中 $N_i$ 是智能体 $i$ 的邻居集合。
3.  Update velocity: 

中文说明：更新速度：

$$
\vec{v}_{new} = \vec{v} + \alpha(\vec{v}_{avg} - \vec{v}), \text{ where } \alpha = 0.05
$$

4.  Update position: 

中文说明：更新位置：

$$
\vec{p}_{new} = \vec{p} + \vec{v}_{new}
$$

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted) / 只能使用原生功能（不得使用外部库）
- The `solve` function signature must remain unchanged / `solve` 函数签名必须保持不变
- The final result must be stored in the `agents_next` array / 最终结果必须存储在 `agents_next` 数组中

## Example 1 / 示例 1

    Input: N = 2
    agents = [
      0.0, 0.0, 1.0, 0.0,    // Agent 0: [x, y, vx, vy]
      3.0, 4.0, 0.0, -1.0    // Agent 1: [x, y, vx, vy]
    ]

    Output:
    agents_next = [
      1.0, 0.0, 1.0, 0.0,    // Agent 0: [x, y, vx, vy]
      3.0, 3.0, 0.0, -1.0    // Agent 1: [x, y, vx, vy]
    ]

## Constraints / 约束

- 1 ≤ `N` ≤ 100,000 / `N` 的范围为 1 至 100,000
- Each agent's position and velocity components are 32-bit floats / 每个智能体的位置和速度分量均为 32 位浮点数
- Performance is measured with `N` = 10,000 / 性能测试使用 `N` = 10,000
