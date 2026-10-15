"""
Model monitoring utilities for OvoScan.

Tracks prediction distributions, confidence scores, and data drift
for the YOLOv8 classification model. Metrics are exposed via
Prometheus for real-time monitoring.
"""
import time
from collections import deque

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

# --- Prometheus Metrics ---

# Prediction metrics
prediction_count = Counter(
    "ovoscan_predictions_total",
    "Total number of predictions made",
    ["class_label"],
)

prediction_confidence = Histogram(
    "ovoscan_prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

prediction_latency = Histogram(
    "ovoscan_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

# Model health metrics
model_loaded = Gauge(
    "ovoscan_model_loaded",
    "Whether the model is loaded (1) or not (0)",
)

model_inference_errors = Counter(
    "ovoscan_inference_errors_total",
    "Total number of inference errors",
)

# Data drift metrics
prediction_class_distribution = Gauge(
    "ovoscan_class_distribution",
    "Current distribution of predicted classes",
    ["class_label"],
)

# PSI drift: compares the live prediction distribution against the distribution
# observed on the validation set. A counter alone cannot detect drift because it
# has no baseline to compare against.
drift_psi = Gauge(
    "ovoscan_prediction_drift_psi",
    "Population Stability Index of the live prediction distribution vs the "
    "validation baseline",
)

drift_status = Gauge(
    "ovoscan_drift_status",
    "Drift interpretation: 0=stable, 1=moderate, 2=significant, -1=unknown",
)

# Rolling window for drift detection
class PredictionTracker:
    """Tracks recent predictions for drift detection."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions: deque = deque(maxlen=window_size)
        self.confidences: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)

    def record(self, prediction: str, confidence: float):
        """Record a prediction for drift analysis."""
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.timestamps.append(time.time())

    def get_class_distribution(self) -> dict:
        """Get the distribution of predictions in the current window."""
        if not self.predictions:
            return {}
        from collections import Counter
        counts = Counter(self.predictions)
        total = len(self.predictions)
        return {cls: count / total for cls, count in counts.items()}

    def get_avg_confidence(self) -> float:
        """Get average confidence in the current window."""
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)

    def get_prediction_rate(self) -> float:
        """Get predictions per second in the current window."""
        if len(self.timestamps) < 2:
            return 0.0
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span == 0:
            return 0.0
        return len(self.predictions) / time_span

    def reset(self):
        """Clear all tracked predictions."""
        self.predictions.clear()
        self.confidences.clear()
        self.timestamps.clear()


# Global tracker instance
tracker = PredictionTracker(window_size=100)


# --- PSI drift against the validation baseline ---

# Baseline class distribution measured on the validation split. Overridable so
# a retrained model can ship its own baseline without a code change.
DEFAULT_BASELINE = {"fertile": 0.5, "infertile": 0.5}
_baseline_distribution: dict[str, float] = dict(DEFAULT_BASELINE)


def set_drift_baseline(distribution: dict[str, float]) -> None:
    """Set the reference class distribution used for PSI.

    Args:
        distribution: Mapping of class label to its proportion in the baseline.
            Proportions are normalised, so raw counts are also accepted.
    """
    global _baseline_distribution

    total = sum(distribution.values())
    if total <= 0:
        return
    _baseline_distribution = {k: v / total for k, v in distribution.items()}


def get_drift_baseline() -> dict[str, float]:
    """Return the current reference distribution."""
    return dict(_baseline_distribution)


def compute_drift_psi(min_samples: int = 30) -> float | None:
    """PSI between the live prediction distribution and the baseline.

    Returns None when there are too few observations to be meaningful. PSI on a
    handful of predictions is noise, not signal.
    """
    from src.utils.drift import compute_psi

    live = tracker.get_class_distribution()
    if not live or len(tracker.predictions) < min_samples:
        return None

    # Compare over the union of labels so a class that disappears entirely still
    # contributes to the index rather than being silently ignored.
    labels = sorted(set(live) | set(_baseline_distribution))
    baseline_values = np.array([_baseline_distribution.get(c, 0.0) for c in labels])
    live_values = np.array([live.get(c, 0.0) for c in labels])

    # PSI needs a distribution, not a single proportion, so expand each class
    # proportion into a weighted sample. This keeps the standard PSI formula
    # while comparing categorical distributions.
    n = 1000
    baseline_sample = np.repeat(np.arange(len(labels)), (baseline_values * n).round().astype(int))
    live_sample = np.repeat(np.arange(len(labels)), (live_values * n).round().astype(int))

    if baseline_sample.size == 0 or live_sample.size == 0:
        return None
    return compute_psi(baseline_sample, live_sample, bins=max(2, len(labels)))


def update_drift_metrics(min_samples: int = 30) -> float | None:
    """Recompute PSI and publish it to Prometheus. Returns the PSI or None."""
    from src.utils.drift import classify_psi

    psi = compute_drift_psi(min_samples=min_samples)
    if psi is None:
        drift_status.set(-1)
        return None

    drift_psi.set(psi)
    drift_status.set({"stable": 0, "moderate": 1, "significant": 2}.get(classify_psi(psi), -1))
    return psi


def record_prediction(prediction: str, confidence: float, latency: float):
    """
    Record a prediction and update all metrics.

    Args:
        prediction: The predicted class label
        confidence: Confidence score (0-1)
        latency: Inference latency in seconds
    """
    prediction_count.labels(class_label=prediction).inc()
    prediction_confidence.observe(confidence)
    prediction_latency.observe(latency)

    tracker.record(prediction, confidence)

    # Update class distribution gauge
    dist = tracker.get_class_distribution()
    for cls, ratio in dist.items():
        prediction_class_distribution.labels(class_label=cls).set(ratio)

    # Recompute PSI against the baseline. Cheap (a rolling window of 100), and
    # doing it here means the gauge is current without a separate poll.
    update_drift_metrics()


def record_inference_error():
    """Record an inference error."""
    model_inference_errors.inc()


def set_model_loaded(loaded: bool):
    """Set the model loaded gauge."""
    model_loaded.set(1 if loaded else 0)


def get_monitoring_summary() -> dict:
    """Get a summary of current monitoring state."""
    from src.utils.drift import classify_psi

    psi = compute_drift_psi()
    return {
        "prediction_rate_per_sec": round(tracker.get_prediction_rate(), 2),
        "avg_confidence": round(tracker.get_avg_confidence(), 4),
        "class_distribution": tracker.get_class_distribution(),
        "window_size": tracker.window_size,
        "drift_psi": round(psi, 6) if psi is not None else None,
        "drift_status": classify_psi(psi) if psi is not None else "insufficient_data",
        "drift_baseline": get_drift_baseline(),
        "total_predictions": sum(
            prediction_count.labels(class_label=c)._value.get()
            for c in ["fertile", "defect"]
        ) if prediction_count._metrics else 0,
    }
