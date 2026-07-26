# Hard Challenges

本目录包含 13 道 GPU 算子题，按主要实现模式大致分为以下几类：

| 类型 | 题目 |
| --- | --- |
| Attention 与序列建模 | `12_multi_head_attention`、`53_casual_attention`（Causal Attention）、`56_linear_attention`、`59_sliding_window_attn` |
| 完整 Transformer 模块 | `74_gpt2_block`、`93_llama_transformer_block` |
| 排序算法 | `15_sorting`、`36_radix_sort` |
| 图搜索与最短路 | `46_bfs_shortest_path`、`73_all_pairs_shortest_paths` |
| 信号处理与频域变换 | `39_Fast_Fourier_transform` |
| 经典机器学习与聚类 | `20_kmeans_clustering` |
| 多智能体与邻域交互模拟 | `14_multi_agent_sim` |

这些题目通常需要处理跨线程协作、迭代依赖、全局同步或复杂的数据访问模式；题目目录按 `<编号>_<名称>` 命名，每题包含题目说明、参考实现和对应的 CUDA/Triton starter 模板。
