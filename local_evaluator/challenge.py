from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DIFFICULTIES = ("easy", "medium", "hard")
LANGUAGE_SOURCES = {
    "cuda": "starter.cu",
    "triton": "starter.triton.py",
}


class ConfigurationError(Exception):
    """Invalid challenge selection or unsupported local configuration."""


@dataclass(frozen=True)
class ChallengeTarget:
    difficulty: str
    number: int
    directory: Path
    source: Path
    language: str

    @property
    def identifier(self) -> str:
        return f"{self.difficulty}/{self.number}"


def resolve_target(repo_root: Path, selector: str, language: str) -> ChallengeTarget:
    match = re.fullmatch(r"(easy|medium|hard)/(\d+)", selector.lower())
    if not match:
        raise ConfigurationError(
            f"Invalid challenge '{selector}'. Use a selector such as easy/1."
        )

    difficulty, number_text = match.groups()
    number = int(number_text)
    matches = sorted((repo_root / "challenges" / difficulty).glob(f"{number}_*"))
    if not matches:
        raise ConfigurationError(f"Challenge {difficulty}/{number} was not found.")
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches)
        raise ConfigurationError(f"Challenge {difficulty}/{number} is ambiguous: {names}")

    source_name = LANGUAGE_SOURCES[language]
    source = matches[0] / "starter" / source_name
    if not source.is_file():
        raise ConfigurationError(
            f"Challenge {difficulty}/{number} does not support {language}: "
            f"missing starter/{source_name}."
        )

    return ChallengeTarget(difficulty, number, matches[0], source, language)


def load_challenge(repo_root: Path, challenge_dir: Path):
    challenges_root = repo_root / "challenges"
    challenges_root_text = str(challenges_root)
    if challenges_root_text not in sys.path:
        sys.path.insert(0, challenges_root_text)

    challenge_file = challenge_dir / "challenge.py"
    module_hash = hashlib.sha256(str(challenge_file).encode()).hexdigest()[:12]
    module_name = f"_leetgpu_challenge_{module_hash}"
    spec = importlib.util.spec_from_file_location(module_name, challenge_file)
    if spec is None or spec.loader is None:
        raise ConfigurationError(f"Cannot load {challenge_file}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    challenge_type = getattr(module, "Challenge", None)
    if challenge_type is None:
        raise ConfigurationError(f"{challenge_file} does not define Challenge.")
    return challenge_type(device="cuda")


def normalize_cases(value: Any, suite_name: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ConfigurationError(
        f"{suite_name} test generator must return a dict or a list of dicts."
    )


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "int8": torch.int8,
    "int32": torch.int32,
    "uint8": torch.uint8,
}


def materialize_value(value: Any, device: str = "cuda") -> Any:
    if isinstance(value, torch.Tensor):
        return value

    descriptor_name = type(value).__name__
    dtype_name = getattr(value, "dtype", None)
    dtype = DTYPES.get(dtype_name)
    if descriptor_name in {"RandTensor", "RandnTensor", "RandIntTensor", "FullTensor", "OutTensor"}:
        if dtype is None:
            raise ConfigurationError(f"Unsupported tensor dtype '{dtype_name}'.")
        shape = tuple(value.shape)
        if descriptor_name == "RandTensor":
            return torch.empty(shape, device=device, dtype=dtype).uniform_(value.low, value.high)
        if descriptor_name == "RandnTensor":
            tensor = torch.empty(shape, device=device, dtype=dtype)
            return tensor.normal_(value.mean, value.std)
        if descriptor_name == "RandIntTensor":
            return torch.randint(value.low, value.high, shape, device=device, dtype=dtype)
        if descriptor_name == "FullTensor":
            return torch.full(shape, value.value, device=device, dtype=dtype)
        return torch.empty(shape, device=device, dtype=dtype)

    return value


def materialize_case(test_case: dict[str, Any]) -> dict[str, Any]:
    return {name: materialize_value(value) for name, value in test_case.items()}


def clone_case(test_case: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in test_case.items()
    }


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
