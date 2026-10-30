"""Tests for the ONNX export helper.

The export itself needs ultralytics and a checkpoint, so these tests cover the
parts that can run without them: argument handling and the benchmark contract.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.edge import export_onnx


def test_benchmark_requires_existing_onnx(tmp_path):
    """A missing artifact must raise rather than silently report zeros.

    The specific exception depends on which dependency fails first, so the
    assertion is that *some* exception propagates instead of a bogus result.
    """
    import onnxruntime as ort

    with pytest.raises((FileNotFoundError, ValueError, RuntimeError, ort.capi.onnxruntime_pybind11_state.Fail)):
        export_onnx.benchmark("missing.pt", str(tmp_path / "nope.onnx"), imgsz=64, runs=1)


def test_benchmark_pins_both_models_to_cpu():
    """The benchmark must compare like-for-like, so device is reported as cpu."""
    import inspect

    src = inspect.getsource(export_onnx.benchmark)
    assert 'device="cpu"' in src
    assert '"device": "cpu"' in src
