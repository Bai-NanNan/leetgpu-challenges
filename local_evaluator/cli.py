from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .backends import BuildError, BuildResult, build_cuda
from .challenge import ConfigurationError, resolve_target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and evaluate a CUDA or Triton starter locally."
    )
    parser.add_argument("challenge", nargs="?", help="Challenge selector, for example easy/1")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"))
    parser.add_argument("--number", "--No", dest="number", type=int)
    parser.add_argument("--lang", "--language", required=True, type=str.lower, choices=("cuda", "triton"))
    parser.add_argument("--case", default="all", choices=("example", "functional", "all"))
    parser.add_argument("--bench", action="store_true", help="Run the performance case after correctness passes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true", help="Ignore the CUDA compilation cache")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _selector(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.challenge:
        if args.difficulty is not None or args.number is not None:
            parser.error("Use either CHALLENGE or --difficulty/--number, not both.")
        return args.challenge
    if args.difficulty is None or args.number is None:
        parser.error("Provide CHALLENGE (for example easy/1) or both --difficulty and --number.")
    return f"{args.difficulty}/{args.number}"


def _print_build(language: str, build: BuildResult | None) -> None:
    if language == "triton":
        print("Build       READY  Triton starter will JIT during execution")
        return
    assert build is not None
    cache = "cache" if build.cached else f"{build.elapsed_seconds:.2f}s"
    print(f"Build       PASS   {cache}")


def _print_failure(failure: dict[str, Any]) -> None:
    stage = failure.get("stage", "unknown")
    message = failure.get("message") or failure.get("reason", "evaluation failed")
    print(f"Failure     {stage.upper()}  {message}")
    for key in ("case", "argument", "index", "expected", "actual", "max_absolute_error", "max_relative_error"):
        if key in failure:
            print(f"  {key}: {failure[key]}")


def _print_result(result: dict[str, Any], bench_requested: bool, verbose: bool) -> None:
    challenge = result.get("challenge", {})
    if challenge:
        print(f"Challenge   {challenge.get('id')} {challenge.get('name')}")
    device = result.get("device", {})
    if device:
        print(
            "Backend     "
            f"{result.get('language', '').upper()} / {device.get('gpu')} / {device.get('compute_capability')}"
        )
        print(
            f"Runtime     CUDA {device.get('cuda')} / PyTorch {device.get('pytorch')} / driver {device.get('driver')}"
        )

    for suite_name in ("example", "functional"):
        suite = result.get("suites", {}).get(suite_name)
        if suite:
            suite_status = "PASS" if suite["passed"] == suite["total"] else "FAIL"
            print(f"{suite_name.title():<11} {suite_status:<6} {suite['passed']}/{suite['total']}")

    for warning in result.get("warnings", []):
        print(f"Warning     {warning}")

    benchmarks = result.get("benchmark", [])
    if bench_requested and benchmarks:
        for index, benchmark in enumerate(benchmarks, start=1):
            prefix = "Benchmark" if len(benchmarks) == 1 else f"Benchmark[{index}]"
            print(
                f"{prefix:<11} median {benchmark['median_ms']:.4f} ms  "
                f"min {benchmark['min_ms']:.4f} ms  p20 {benchmark['p20_ms']:.4f} ms  "
                f"p80 {benchmark['p80_ms']:.4f} ms"
            )
            print(
                f"  warmup: {benchmark['warmup_iterations']} iterations, "
                f"batch: {benchmark['batch_size']}"
            )

    if result.get("status") == "pass":
        print("Result      PASS")
    else:
        _print_failure(result.get("failure", {}))
        print("Result      FAIL")
        if verbose and result.get("failure", {}).get("traceback"):
            print(result["failure"]["traceback"])


def main(argv: list[str] | None = None, repo_root: Path | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero.")
    selector = _selector(args, parser)
    root = repo_root or Path(__file__).resolve().parents[1]

    try:
        target = resolve_target(root, selector, args.lang)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    build: BuildResult | None = None
    if target.language == "cuda":
        try:
            build = build_cuda(target.source, root / ".leetgpu" / "cache", target.identifier, args.rebuild)
        except BuildError as exc:
            print("Build       FAIL")
            print(str(exc), file=sys.stderr)
            return 1

    _print_build(target.language, build)
    if args.verbose and build is not None and build.output:
        print(build.output)

    job = {
        "repo_root": str(root),
        "challenge_dir": str(target.directory),
        "challenge_id": target.identifier,
        "language": target.language,
        "source": str(target.source),
        "artifact": str(build.artifact) if build else None,
        "case": args.case,
        "bench": args.bench,
        "seed": args.seed,
        "verbose": args.verbose,
    }
    with tempfile.TemporaryDirectory(prefix="leetgpu-eval-") as temporary_directory:
        temporary = Path(temporary_directory)
        job_path = temporary / "job.json"
        result_path = temporary / "result.json"
        job_path.write_text(json.dumps(job))
        command = [
            sys.executable,
            "-m",
            "local_evaluator.worker",
            "--job",
            str(job_path),
            "--result",
            str(result_path),
        ]
        worker_environment = os.environ.copy()
        evaluator_root = str(Path(__file__).resolve().parents[1])
        existing_python_path = worker_environment.get("PYTHONPATH")
        worker_environment["PYTHONPATH"] = (
            f"{evaluator_root}{os.pathsep}{existing_python_path}"
            if existing_python_path
            else evaluator_root
        )
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=worker_environment,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"Failure     TIMEOUT  worker exceeded {args.timeout:g} seconds")
            print("Result      FAIL")
            return 1

        if not result_path.is_file():
            print(f"Failure     WORKER  exited with status {completed.returncode}")
            if completed.stderr:
                print(completed.stderr, file=sys.stderr)
            print("Result      FAIL")
            return 1
        result = json.loads(result_path.read_text())
        _print_result(result, args.bench, args.verbose)
        if args.verbose:
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
        return 0 if result.get("status") == "pass" else 1
