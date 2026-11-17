#!/usr/bin/env python3
"""OvoScan Top-1 accuracy with a bootstrap confidence interval.

The 20-image test split ships with empty label files, so it is blind and cannot
be scored locally. This evaluates the 40-image validation set instead.

Two things here are easy to get wrong and are worth knowing about:

1. **Class-index inversion.** The served checkpoint reports
   ``{0: 'defect', 1: 'fertile'}`` while ``egg_dataset/data.yaml`` declares
   ``['fertile', 'infertile']``. The orderings are opposite, so comparing the
   raw index scores 0/40 — every prediction counted wrong. ``_map_prediction``
   inverts before comparing.

2. **Inference is the expensive part, not the resampling.** Running the model
   inside the bootstrap loop means 40,000 forward passes for 1000 iterations.
   Predictions are computed once and the bootstrap resamples the cached
   correctness vector, which is mathematically identical and far faster.

Run as a script for the full report, or let pytest collect the tests:

    python tests/test_model_quality.py --model serving/model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "egg_dataset"
DEFAULT_MODEL = ROOT / "serving" / "model.pt"

# The checkpoint's class order is the reverse of data.yaml's.
INVERT_CLASS_INDEX = True


def load_val_data(data_dir: Path = DEFAULT_DATA) -> List[Tuple[str, int]]:
    """Return (image_path, label) pairs from the validation split.

    Labels are YOLO detection files; the first token is the class index. Only
    the first line is read because the task is classification — one label per
    image.
    """
    images_dir = data_dir / "valid" / "images"
    labels_dir = data_dir / "valid" / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"validation images not found at {images_dir}")

    samples: List[Tuple[str, int]] = []
    for img in sorted(images_dir.glob("*.jpg")):
        label_file = labels_dir / f"{img.stem}.txt"
        if not label_file.exists():
            continue
        first = label_file.read_text().split("\n", 1)[0].strip()
        if not first:
            continue
        samples.append((str(img), int(first.split()[0])))
    return samples


def _map_prediction(raw_index: int) -> int:
    """Translate the checkpoint's class index into the dataset's ordering."""
    return 1 - raw_index if INVERT_CLASS_INDEX else raw_index


def predict_correctness(
    model_path: Path, samples: Sequence[Tuple[str, int]]
) -> np.ndarray:
    """Return a 0/1 array marking whether each sample was classified correctly."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    out = np.zeros(len(samples), dtype=np.int64)
    for i, (img_path, true_label) in enumerate(samples):
        raw = int(model(img_path, verbose=False)[0].probs.top1)
        out[i] = int(_map_prediction(raw) == true_label)
    return out


def bootstrap_ci(
    correctness: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Percentile bootstrap over a cached correctness vector.

    Returns (point_estimate, ci_lower, ci_upper) as fractions in [0, 1].
    """
    if len(correctness) == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    n = len(correctness)
    idx = rng.integers(0, n, size=(n_iterations, n))
    accuracies = correctness[idx].mean(axis=1)

    alpha = (1 - confidence) / 2
    return (
        float(correctness.mean()),
        float(np.percentile(accuracies, 100 * alpha)),
        float(np.percentile(accuracies, 100 * (1 - alpha))),
    )


# --- tests -----------------------------------------------------------------


def test_class_index_inversion_is_required():
    """The raw index scores 0/40; the inverted index scores 40/40.

    This is the bug that makes a perfect model look broken. If someone removes
    the inversion, this test fails rather than the README silently becoming
    wrong.

    The dataset is gitignored (it is 139 training images of licensed data), so
    this skips in CI and runs locally where the data exists.
    """
    import pytest

    if not (DEFAULT_DATA / "valid" / "images").exists():
        pytest.skip("egg_dataset/ is not present (gitignored); run locally")

    samples = load_val_data()
    assert len(samples) == 40, f"expected 40 validation samples, got {len(samples)}"

    from ultralytics import YOLO

    model = YOLO(str(DEFAULT_MODEL))
    raw_correct = 0
    mapped_correct = 0
    for img_path, true_label in samples:
        raw = int(model(img_path, verbose=False)[0].probs.top1)
        raw_correct += int(raw == true_label)
        mapped_correct += int(_map_prediction(raw) == true_label)

    assert raw_correct == 0, (
        f"raw (non-inverted) index scored {raw_correct}/40; the inversion may no "
        "longer be needed, or the checkpoint changed"
    )
    assert mapped_correct == 40, (
        f"mapped index scored {mapped_correct}/40; _map_prediction is not "
        "inverting, so every prediction is counted wrong"
    )


def test_bootstrap_ci_brackets_point_estimate():
    """A CI must contain its own point estimate."""
    correctness = np.array([1] * 36 + [0] * 4)
    point, lo, hi = bootstrap_ci(correctness, n_iterations=500)
    assert lo <= point <= hi, f"CI [{lo}, {hi}] excludes point {point}"
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_is_deterministic():
    """Same seed, same interval — otherwise the README number drifts."""
    correctness = np.array([1, 1, 0, 1, 1, 1, 0, 1])
    a = bootstrap_ci(correctness, n_iterations=200, seed=7)
    b = bootstrap_ci(correctness, n_iterations=200, seed=7)
    assert a == b


def test_perfect_sample_collapses_interval():
    """40/40 gives [1.0, 1.0] — the interval is the honest part of the claim."""
    point, lo, hi = bootstrap_ci(np.ones(40, dtype=np.int64), n_iterations=500)
    assert (point, lo, hi) == (1.0, 1.0, 1.0)


def test_empty_sample_raises():
    """Guarding the degenerate case keeps the failure loud."""
    try:
        bootstrap_ci(np.array([], dtype=np.int64))
    except ValueError:
        return
    raise AssertionError("expected ValueError for an empty sample")


# --- script ----------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--out", default="bootstrap_results.json")
    args = ap.parse_args()

    samples = load_val_data(Path(args.data))
    print(f"Loaded {len(samples)} validation samples")

    correctness = predict_correctness(Path(args.model), samples)
    point, lo, hi = bootstrap_ci(
        correctness, n_iterations=args.iterations, confidence=args.confidence
    )

    print(f"\nBootstrap ({args.iterations} iterations, {args.confidence:.0%} CI)")
    print(f"  Correct:        {int(correctness.sum())}/{len(correctness)}")
    print(f"  Point estimate: {point:.4f} ({point:.1%})")
    print(f"  95% CI:         [{lo:.4f}, {hi:.4f}] ({lo:.1%} - {hi:.1%})")

    Path(args.out).write_text(
        json.dumps(
            {
                "point_estimate": point,
                "ci_lower": lo,
                "ci_upper": hi,
                "confidence": args.confidence,
                "n_iterations": args.iterations,
                "n_samples": len(correctness),
                "n_correct": int(correctness.sum()),
                "split": "valid",
            },
            indent=2,
        )
    )
    print(f"\nResults saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
