# LeetGPU

This is the challenge set for [LeetGPU.com](https://leetgpu.com). We welcome contributions and bug reports!

## Overview

Each challenge includes problem descriptions, reference implementation, test cases, and starter templates for multiple GPU programming frameworks.

## Challenge Structure

Each challenge contains:

- **`challenge.html`**: Detailed problem description, examples, and constraints
- **`challenge.py`**: Reference implementation, test cases, and challenge metadata
- **`starter/`**: Template files for each supported framework

## 本地 CUDA/Triton 测评

项目提供本地测评入口，用于练习和优化每道题的 CUDA 与 Triton starter。直接修改题目目录中的 starter 文件：

- CUDA：`starter/starter.cu`
- Triton：`starter/starter.triton.py`

先创建本地环境：

```bash
uv sync --group test --managed-python --python 3.12
```

默认命令会编译（CUDA）或导入（Triton）starter，并运行样例与功能测试：

```bash
uv run python evaluate.py easy/1 --lang cuda
uv run python evaluate.py easy/1 --lang triton
```

使用 `--bench` 在正确性全部通过后，运行性能输入、warmup 和 CUDA Event 计时：

```bash
uv run python evaluate.py easy/1 --lang cuda --bench
```

也可使用 `--difficulty easy --number 1`。常用选项包括 `--case example|functional|all`、`--seed 0`、`--rebuild`、`--timeout 120` 和 `--verbose`。

CUDA 编译产物缓存于 `.leetgpu/cache/`，修改 starter、NVCC 版本或本机 GPU 架构变化后会自动重新编译。测评只比较本机不同实现的性能，不等价于远程 T4 排名。当前不支持缺少 CUDA/Triton starter 的题目，例如 `easy/41_simple_inference`。

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing new challenges or improvements.

## License

This problem set is licensed under [CC BY‑NC‑ND 4.0 license](LICENSE).

© 2025 AlphaGPU, LLC. Commercial use, redistribution, or derivative use is prohibited.
