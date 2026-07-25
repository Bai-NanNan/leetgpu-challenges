from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


class BuildError(Exception):
    """Candidate source could not be compiled."""


class RuntimeBackendError(Exception):
    """Candidate source could not be loaded or invoked."""


@dataclass(frozen=True)
class BuildResult:
    artifact: Path
    cached: bool
    elapsed_seconds: float
    command: list[str]
    output: str


def _nvcc_version(nvcc: str) -> str:
    completed = subprocess.run(
        [nvcc, "--version"], capture_output=True, text=True, check=False
    )
    return f"{completed.stdout}\n{completed.stderr}".strip()


def build_cuda(
    source: Path,
    cache_root: Path,
    challenge_id: str,
    rebuild: bool = False,
) -> BuildResult:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise BuildError("nvcc was not found in PATH.")
    if not torch.cuda.is_available():
        raise BuildError("PyTorch cannot access a CUDA device.")

    major, minor = torch.cuda.get_device_capability()
    architecture = f"sm_{major}{minor}"
    flags = [
        "-O3",
        "-std=c++17",
        "--shared",
        "-Xcompiler=-fPIC",
        "-lineinfo",
        f"-arch={architecture}",
    ]
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update("\0".join(flags).encode())
    digest.update(_nvcc_version(nvcc).encode())
    cache_key = digest.hexdigest()[:20]
    artifact_dir = cache_root / challenge_id.replace("/", "_") / cache_key
    artifact = artifact_dir / "starter.so"
    command = [nvcc, *flags, str(source), "-o", str(artifact)]

    if artifact.is_file() and not rebuild:
        return BuildResult(artifact, True, 0.0, command, "")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    if completed.returncode != 0 or not artifact.is_file():
        raise BuildError(output or f"nvcc exited with status {completed.returncode}.")
    return BuildResult(artifact, False, elapsed, command, output)


class CudaBackend:
    def __init__(self, artifact: Path, signature: dict[str, tuple[Any, str]]):
        try:
            self.library = ctypes.CDLL(str(artifact))
            self.solve = self.library.solve
        except (OSError, AttributeError) as exc:
            raise RuntimeBackendError(f"Cannot load CUDA solve symbol: {exc}") from exc

        self.signature = signature
        self.solve.argtypes = [parameter_type for parameter_type, _ in signature.values()]
        self.solve.restype = None

    def invoke(self, arguments: dict[str, Any]) -> None:
        values = []
        for name, (parameter_type, _) in self.signature.items():
            value = arguments[name]
            if isinstance(value, torch.Tensor):
                pointer = ctypes.c_void_p(value.data_ptr())
                values.append(ctypes.cast(pointer, parameter_type))
            else:
                values.append(parameter_type(value))
        self.solve(*values)


class TritonBackend:
    def __init__(self, source: Path, signature: dict[str, tuple[Any, str]]):
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        module_name = f"_leetgpu_triton_{source_hash}"
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise RuntimeBackendError(f"Cannot import Triton source {source}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise RuntimeBackendError(f"Triton import failed: {exc}") from exc
        solve = getattr(module, "solve", None)
        if not callable(solve):
            raise RuntimeBackendError("Triton starter does not define a callable solve().")
        self.solve = solve
        self.argument_names = list(signature)

    def invoke(self, arguments: dict[str, Any]) -> None:
        self.solve(*(arguments[name] for name in self.argument_names))
