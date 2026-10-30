"""Export the OvoScan YOLOv8 detector to ONNX and benchmark it.

ONNX Runtime is the portable edge target: it runs on CPU, CUDA and TensorRT
execution providers from one artifact, so the same exported file serves a
laptop demo and a GPU edge box. This module exports the trained checkpoint and
measures the latency difference against the PyTorch model, which is the number
that matters for the 30 FPS / <35 ms requirement.

Usage:
    python -m src.edge.export_onnx --model serving/model.pt --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def export(model_path: str, imgsz: int = 640, opset: int = 12) -> str:
    """Export a YOLO checkpoint to ONNX and return the artifact path."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    # simplify=True folds redundant ops; dynamic=False keeps a fixed batch of 1,
    # which is what a single-image edge inference path uses and what TensorRT
    # can optimise hardest.
    out = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False)
    return str(out)


def benchmark(model_path: str, onnx_path: str, imgsz: int = 640, runs: int = 20) -> dict:
    """Compare PyTorch and ONNX Runtime latency on the same random input.

    Both models are pinned to CPU. Comparing a CUDA PyTorch run against a CPU
    ONNX Runtime run measures the hardware, not the export: the ONNX artifact
    looks slower purely because onnxruntime-gpu is not installed. Pinning both
    to CPU makes the number an honest measure of the export itself.
    """
    import onnxruntime as ort
    from ultralytics import YOLO

    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)

    torch_model = YOLO(model_path)
    # Warm up so the first-call graph build is not counted in the measurement.
    torch_model(image, verbose=False, device="cpu")
    t0 = time.perf_counter()
    for _ in range(runs):
        torch_model(image, verbose=False, device="cpu")
    torch_ms = (time.perf_counter() - t0) / runs * 1000

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    blob = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    session.run(None, {input_name: blob})
    t0 = time.perf_counter()
    for _ in range(runs):
        session.run(None, {input_name: blob})
    onnx_ms = (time.perf_counter() - t0) / runs * 1000

    return {
        "imgsz": imgsz,
        "runs": runs,
        "device": "cpu",
        "torch_latency_ms": round(torch_ms, 2),
        "onnx_latency_ms": round(onnx_ms, 2),
        "speedup": round(torch_ms / onnx_ms, 2) if onnx_ms else None,
        "onnx_size_mb": round(os.path.getsize(onnx_path) / 1e6, 2),
        "onnx_providers": ort.get_available_providers(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OvoScan YOLOv8 to ONNX and benchmark")
    parser.add_argument("--model", default="serving/model.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    onnx_path = args.out or export(args.model, args.imgsz)
    print(f"Exported: {onnx_path}")
    print(json.dumps(benchmark(args.model, onnx_path, args.imgsz, args.runs), indent=2))


if __name__ == "__main__":
    main()
