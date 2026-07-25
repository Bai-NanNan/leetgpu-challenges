import shutil
from pathlib import Path

import pytest
import torch

from local_evaluator.cli import main


ROOT = Path(__file__).resolve().parents[2]

CUDA_SOURCE = r'''
#include <cuda_runtime.h>

__global__ void add(const float* a, const float* b, float* c, size_t n) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) c[index] = a[index] + b[index];
}

extern "C" void solve(const float* a, const float* b, float* c, size_t n) {
    add<<<(n + 255) / 256, 256>>>(a, b, c, n);
}
'''

TRITON_SOURCE = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(a, b, c, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(c + offsets, tl.load(a + offsets, mask=mask) + tl.load(b + offsets, mask=mask), mask=mask)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, n: int):
    add_kernel[(triton.cdiv(n, 256),)](a, b, c, n, BLOCK=256)
'''


@pytest.fixture
def temporary_vector_repo(tmp_path: Path) -> Path:
    challenge_root = tmp_path / "challenges"
    (challenge_root / "core").mkdir(parents=True)
    shutil.copy2(ROOT / "challenges/core/challenge_base.py", challenge_root / "core/challenge_base.py")
    target = challenge_root / "easy/1_vector_add"
    (target / "starter").mkdir(parents=True)
    shutil.copy2(ROOT / "challenges/easy/1_vector_add/challenge.py", target / "challenge.py")
    return tmp_path


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_starter_passes_in_temporary_repo(temporary_vector_repo: Path):
    source = temporary_vector_repo / "challenges/easy/1_vector_add/starter/starter.cu"
    source.write_text(CUDA_SOURCE)

    assert main(["easy/1", "--lang", "cuda"], repo_root=temporary_vector_repo) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_starter_passes_in_temporary_repo(temporary_vector_repo: Path):
    source = temporary_vector_repo / "challenges/easy/1_vector_add/starter/starter.triton.py"
    source.write_text(TRITON_SOURCE)

    assert main(["easy/1", "--lang", "triton"], repo_root=temporary_vector_repo) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_wrong_cuda_starter_returns_failure(temporary_vector_repo: Path):
    source = temporary_vector_repo / "challenges/easy/1_vector_add/starter/starter.cu"
    source.write_text(
        '#include <cuda_runtime.h>\nextern "C" void solve(const float*, const float*, float*, size_t) {}\n'
    )

    assert main(["easy/1", "--lang", "cuda", "--case", "example"], repo_root=temporary_vector_repo) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_benchmark_runs_after_correctness(temporary_vector_repo: Path):
    source = temporary_vector_repo / "challenges/easy/1_vector_add/starter/starter.cu"
    source.write_text(CUDA_SOURCE)

    assert main(
        ["easy/1", "--lang", "cuda", "--case", "example", "--bench", "--timeout", "60"],
        repo_root=temporary_vector_repo,
    ) == 0
