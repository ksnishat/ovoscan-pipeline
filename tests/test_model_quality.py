#!/usr/bin/env python3
"""
OvoScan Bootstrap Confidence Interval Evaluation

Computes a 95% confidence interval for the Top-1 accuracy using bootstrap
resampling (1000 iterations) on the VALIDATION set (test set has no labels).
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO


def load_val_data(data_dir: str = "egg_dataset") -> List[Tuple[str, int]]:
    """Load validation set image paths and labels."""
    val_dir = Path(data_dir) / "valid" / "images"
    labels_dir = Path(data_dir) / "valid" / "labels"
    
    if not val_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Validation data not found at {data_dir}/valid")
    
    samples = []
    for img_file in sorted(val_dir.glob("*.jpg")):
        stem = img_file.stem
        label_file = Path(data_dir) / "valid" / "labels" / f"{img_file.stem}.txt"
        if not label_file.exists():
            continue
        with open(label_file) as f:
            first_line = f.readline().strip()
        if not first_line:
            continue
        class_idx = int(first_line.split()[0])
        samples.append((str(img_file), class_idx))
    
    return samples


def evaluate_model(model_path: str, samples: List[Tuple[str, int]]) -> float:
    """Evaluate model on samples and return Top-1 accuracy."""
    model = YOLO(model_path)
    correct = 0
    total = 0
    
    for img_path, true_label in samples:
        results = model(img_path, verbose=False)
        pred = results[0].probs.top1
        
        # Model classes: 0='defect', 1='fertile'
        # Dataset: 0=fertile, 1=infertile(defect)
        # Need to map: model 0->1, 1->0
        pred = results[0].probs.top1
        pred_mapped = 1 - pred  # invert: model 0=defect->1, 1=fertile->0
        
        if pred_mapped == true_label:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def bootstrap_ci(
    model_path: str,
    samples: List[Tuple[str, int]],
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for Top-1 accuracy.
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(seed)
    n = len(samples)
    accuracies = []
    
    model = YOLO(model_path)
    
    for i in range(n_iterations):
        # Bootstrap sample with replacement
        indices = np.random.choice(len(samples), size=len(samples), replace=True)
        bootstrap_samples = [samples[i] for i in indices]
        
        correct = 0
        for img_path, true_label in bootstrap_samples:
            results = model(img_path, verbose=False)
            pred = results[0].probs.top1
            pred_mapped = 1 - pred  # invert: model 0=defect->1, 1=fertile->0
            if pred_mapped == true_label:
                correct += 1
        
        acc = correct / len(bootstrap_samples)
        accuracies.append(acc)
    
    point_estimate = np.mean(accuracies)
    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(accuracies, 100 * alpha)
    ci_upper = np.percentile(accuracies, 100 * (1 - alpha))
    
    return point_estimate, ci_lower, ci_upper


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bootstrap CI for OvoScan Top-1 accuracy")
    parser.add_argument("--model", type=str, default="serving/model.pt", help="Path to model weights")
    parser.add_argument("--data-dir", type=str, default="egg_dataset", help="Path to dataset root")
    parser.add_argument("--iterations", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()
    
    # Load validation samples
    samples = load_val_data(args.data_dir)
    print(f"Loaded {len(samples)} validation samples")
    
    if len(samples) == 0:
        print("ERROR: No validation samples found")
        sys.exit(1)
    
    # Run bootstrap
    print(f"Running bootstrap with {args.iterations} iterations...")
    point, ci_lower, ci_upper = bootstrap_ci(
        model_path=args.model,
        samples=samples,
        n_iterations=args.iterations,
        confidence=args.confidence,
        seed=args.seed
    )
    
    # Output
    if args.json:
        result = {
            "point_estimate": point,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence": args.confidence,
            "n_iterations": args.iterations,
            "n_samples": len(load_val_data(args.data_dir)),
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"\nBootstrap Results ({args.iterations} iterations, {args.confidence*100:.0f}% CI):")
        print(f"  Point estimate: {point:.4f} ({point*100:.1f}%)")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] ({ci_lower*100:.1f}% - {ci_upper*100:.1f}%)")
        print(f"  Iterations: {args.iterations}")
        print(f"  Validation samples: {len(load_val_data(args.data_dir))}")
    
    # Save results
    result = {
        "point_estimate": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": 0.95,
        "n_iterations": 1000,
        "n_samples": len(load_val_data("egg_dataset")),
    }
    with open("bootstrap_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nResults saved to bootstrap_results.json")


if __name__ == "__main__":
    main()
