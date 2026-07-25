from __future__ import annotations

import math
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .backends import CudaBackend, RuntimeBackendError, TritonBackend
from .challenge import clone_case, load_challenge, materialize_case, normalize_cases, set_seed


class EvaluationError(Exception):
    """The candidate could not be evaluated safely."""


def _device_metadata() -> dict[str, Any]:
    driver = "unknown"
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode == 0:
            driver = completed.stdout.strip().splitlines()[0]
    except (FileNotFoundError, IndexError):
        pass

    major, minor = torch.cuda.get_device_capability()
    return {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"sm_{major}{minor}",
        "driver": driver,
        "cuda": torch.version.cuda or "unknown",
        "pytorch": torch.__version__,
        "python": platform.python_version(),
    }


def _load_backend(job: dict[str, Any], signature: dict[str, tuple[Any, str]]):
    if job["language"] == "cuda":
        return CudaBackend(Path(job["artifact"]), signature)
    return TritonBackend(Path(job["source"]), signature)


def _comparison_details(
    expected: torch.Tensor, actual: torch.Tensor, atol: float, rtol: float
) -> dict[str, Any]:
    if expected.shape != actual.shape:
        return {"reason": "shape mismatch", "expected_shape": list(expected.shape), "actual_shape": list(actual.shape)}
    if expected.dtype != actual.dtype:
        return {"reason": "dtype mismatch", "expected_dtype": str(expected.dtype), "actual_dtype": str(actual.dtype)}

    if expected.is_floating_point() or expected.is_complex():
        close = torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
        expected_float = expected.to(torch.float64)
        actual_float = actual.to(torch.float64)
        absolute = (actual_float - expected_float).abs()
        relative = absolute / expected_float.abs().clamp_min(torch.finfo(torch.float64).tiny)
    else:
        close = actual == expected
        absolute = (actual.to(torch.int64) - expected.to(torch.int64)).abs().to(torch.float64)
        relative = absolute

    mismatch = (~close).flatten().nonzero()
    index = int(mismatch[0].item()) if mismatch.numel() else 0
    expected_value = expected.flatten()[index].item() if expected.numel() else None
    actual_value = actual.flatten()[index].item() if actual.numel() else None
    return {
        "reason": "values differ",
        "index": index,
        "expected": expected_value,
        "actual": actual_value,
        "max_absolute_error": float(torch.nan_to_num(absolute, nan=math.inf).max().item()),
        "max_relative_error": float(torch.nan_to_num(relative, nan=math.inf).max().item()),
    }


def _assert_close(name: str, expected: torch.Tensor, actual: torch.Tensor, atol: float, rtol: float) -> dict[str, Any] | None:
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
    except AssertionError:
        detail = _comparison_details(expected, actual, atol, rtol)
        detail["argument"] = name
        return detail
    return None


def _run_case(
    challenge: Any,
    backend: Any,
    signature: dict[str, tuple[Any, str]],
    test_case: dict[str, Any],
    label: str,
) -> dict[str, Any] | None:
    original = materialize_case(test_case)
    expected = clone_case(original)
    actual = clone_case(original)

    try:
        challenge.reference_impl(**expected)
        torch.cuda.synchronize()
    except Exception as exc:
        return {"stage": "reference", "case": label, "message": str(exc)}

    try:
        backend.invoke(actual)
        torch.cuda.synchronize()
    except Exception as exc:
        return {"stage": "runtime", "case": label, "message": str(exc)}

    for name, (_, direction) in signature.items():
        if not isinstance(actual[name], torch.Tensor):
            continue
        if direction in {"out", "inout"}:
            mismatch = _assert_close(name, expected[name], actual[name], challenge.atol, challenge.rtol)
            if mismatch is not None:
                mismatch.update({"stage": "correctness", "case": label})
                return mismatch
        elif direction == "in" and not torch.equal(original[name], actual[name]):
            return {
                "stage": "correctness",
                "case": label,
                "argument": name,
                "reason": "input tensor was modified",
            }
    return None


def _run_suite(
    suite_name: str,
    cases: list[dict[str, Any]],
    challenge: Any,
    backend: Any,
    signature: dict[str, tuple[Any, str]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    for index, test_case in enumerate(cases, start=1):
        failure = _run_case(challenge, backend, signature, test_case, f"{suite_name}[{index}]")
        if failure is not None:
            return {"passed": index - 1, "total": len(cases)}, failure
    return {"passed": len(cases), "total": len(cases)}, None


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _case_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for name, value in arguments.items():
        if isinstance(value, torch.Tensor):
            summary[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            summary[name] = value
    return summary


def _benchmark_case(backend: Any, signature: dict[str, tuple[Any, str]], test_case: dict[str, Any]) -> dict[str, Any]:
    arguments = materialize_case(test_case)
    inout = [name for name, (_, direction) in signature.items() if direction == "inout"]
    snapshots = {name: arguments[name].clone() for name in inout if isinstance(arguments[name], torch.Tensor)}

    def restore_inouts() -> None:
        for name, snapshot in snapshots.items():
            arguments[name].copy_(snapshot)

    def invoke() -> None:
        restore_inouts()
        backend.invoke(arguments)

    warmup_started = time.perf_counter()
    warmup_iterations = 0
    while warmup_iterations < 5 or (
        time.perf_counter() - warmup_started < 0.1 and warmup_iterations < 25
    ):
        invoke()
        torch.cuda.synchronize()
        warmup_iterations += 1

    restore_inouts()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    backend.invoke(arguments)
    end.record()
    end.synchronize()
    single_ms = max(start.elapsed_time(end), 0.001)
    batch_size = 1 if snapshots else max(1, min(1000, round(50.0 / single_ms)))

    samples = []
    for _ in range(10):
        restore_inouts()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(batch_size):
            backend.invoke(arguments)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / batch_size)

    return {
        "input": _case_summary(arguments),
        "warmup_iterations": warmup_iterations,
        "batch_size": batch_size,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "p20_ms": _percentile(samples, 0.20),
        "p80_ms": _percentile(samples, 0.80),
    }


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(job["repo_root"])
    challenge_dir = Path(job["challenge_dir"])
    set_seed(job["seed"])
    if not torch.cuda.is_available():
        raise EvaluationError("PyTorch cannot access a CUDA device.")

    challenge = load_challenge(repo_root, challenge_dir)
    signature = challenge.get_solve_signature()
    backend = _load_backend(job, signature)
    result: dict[str, Any] = {
        "status": "pass",
        "challenge": {"id": job["challenge_id"], "name": challenge.name},
        "language": job["language"],
        "device": _device_metadata(),
        "suites": {},
        "warnings": [],
    }

    if (
        job["bench"]
        and job["language"] == "cuda"
        and "cudaDeviceSynchronize" in Path(job["source"]).read_text()
    ):
        result["warnings"].append(
            "starter contains cudaDeviceSynchronize(); benchmark timing may include synchronization overhead."
        )

    suites = []
    if job["case"] in {"example", "all"}:
        suites.append(("example", normalize_cases(challenge.generate_example_test(), "example")))
    if job["case"] in {"functional", "all"}:
        suites.append(("functional", normalize_cases(challenge.generate_functional_test(), "functional")))

    for suite_name, cases in suites:
        summary, failure = _run_suite(suite_name, cases, challenge, backend, signature)
        result["suites"][suite_name] = summary
        if failure is not None:
            result.update({"status": "fail", "failure": failure})
            return result

    if job["bench"]:
        try:
            torch.cuda.empty_cache()
            performance_cases = normalize_cases(challenge.generate_performance_test(), "performance")
            result["benchmark"] = [
                _benchmark_case(backend, signature, test_case) for test_case in performance_cases
            ]
        except Exception as exc:
            result.update(
                {
                    "status": "fail",
                    "failure": {"stage": "benchmark", "message": str(exc)},
                }
            )
    return result
