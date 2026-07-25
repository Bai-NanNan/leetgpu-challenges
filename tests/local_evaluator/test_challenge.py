from pathlib import Path

import pytest
import torch

from local_evaluator.challenge import (
    ConfigurationError,
    materialize_case,
    resolve_target,
)


ROOT = Path(__file__).resolve().parents[2]


def test_resolve_starter_sources():
    cuda = resolve_target(ROOT, "easy/1", "cuda")
    triton = resolve_target(ROOT, "easy/1", "triton")

    assert cuda.source.name == "starter.cu"
    assert triton.source.name == "starter.triton.py"


def test_rejects_unsupported_challenge():
    with pytest.raises(ConfigurationError, match="does not support cuda"):
        resolve_target(ROOT, "easy/41", "cuda")


def test_materializes_descriptor_like_values():
    rand = type("RandTensor", (), {"shape": (4,), "low": -1.0, "high": 1.0, "dtype": "float32"})()
    out = type("OutTensor", (), {"shape": (4,), "dtype": "int32"})()

    case = materialize_case({"x": rand, "out": out, "n": 4})

    assert case["x"].shape == (4,)
    assert case["x"].dtype == torch.float32
    assert case["out"].dtype == torch.int32
    assert case["n"] == 4
