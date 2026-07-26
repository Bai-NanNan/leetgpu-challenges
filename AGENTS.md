# Repository Guidelines

## Project Structure & Module Organization

Challenge content lives in `challenges/<difficulty>/<number>_<name>/`. Each
challenge contains `challenge.py` (reference implementation, signatures, and
test generators), `challenge.html` (the rendered prompt), and `starter/`
templates. Shared challenge types are in `challenges/core/`.

The local CUDA/Triton evaluator starts at `evaluate.py`. Its implementation is
in `local_evaluator/`: keep CLI parsing in `cli.py`, challenge loading and test
materialization in `challenge.py`, and isolated execution in `worker.py`.
Evaluator tests live in `tests/local_evaluator/`. Remote submission and update
utilities remain under `scripts/`.

## Build, Test, and Development Commands

Create the managed Python 3.12 environment with:

```bash
uv sync --group test --managed-python --python 3.12
```

Run evaluator tests with `uv run --group test pytest -q tests/local_evaluator`.
GPU integration tests require a CUDA-capable PyTorch installation and `nvcc`;
run unit-only checks with `uv run --group test pytest -q -m 'not gpu'`.

Exercise a starter directly with `uv run python evaluate.py easy/1 --lang cuda`
or `--lang triton`; add `--bench` only after correctness passes. CUDA build
artifacts are local cache data in `.leetgpu/cache/` and must not be committed.

## Coding Style & Naming Conventions

Target Python 3.12, use four-space indentation, type hints where interfaces are
non-obvious, and Black-compatible formatting with a 100-character line limit.
Sort imports with isort and keep flake8 clean. Format changed Python with
`black challenges/ scripts/` and `isort challenges/ scripts/`; run
`flake8 challenges/ scripts/` before opening a PR. Format CUDA/C++ with
`clang-format` using the repository configuration.

Name challenge directories `<number>_<lowercase_underscore_name>`. Preserve
the exact `solve` ABI declared by `get_solve_signature()` and the established
starter names such as `starter.cu` and `starter.triton.py`.

## Testing Guidelines

Name test modules `test_*.py` and tests `test_<behavior>`. Add focused unit
tests for parsing, materialization, comparison, or cache behavior; mark tests
requiring an actual GPU with `@pytest.mark.gpu`. For challenge changes, ensure
the example, functional cases, and `challenge.html` remain consistent.

## Commit & Pull Request Guidelines

Use concise imperative subjects, following existing history, for example
`Add challenge 106: Token Embedding Layer (Medium)` or `Fix example`. Keep a
commit scoped to one change. PRs should explain the behavioral change, list
validation commands and hardware assumptions, link relevant issues when
available, and include the complete challenge files for new problems. Follow
the contributor terms in `CONTRIBUTING.md`.
