"""
Model monitoring utilities for OvoScan.

Tracks prediction distributions, confidence scores, and data drift
for the YOLOv8 classification model. Metrics are exposed via
Prometheus for real-time monitoring.
"""
import time
from collections import deque
from typing import Optional

from prometheus_client import Counter, Histogram, Gauge, Summary

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


def record_inference_error():
    """Record an inference error."""
    model_inference_errors.inc()


def set_model_loaded(loaded: bool):
    """Set the model loaded gauge."""
    model_loaded.set(1 if loaded else 0)


def get_monitoring_summary() -> dict:
    """Get a summary of current monitoring state."""
    return {
        "prediction_rate_per_sec": round(tracker.get_prediction_rate(), 2),
        "avg_confidence": round(tracker.get_avg_confidence(), 4),
        "class_distribution": tracker.get_class_distribution(),
        "window_size": tracker.window_size,
        "total_predictions": sum(
            prediction_count.labels(class_label=c)._value.get()
            for c in ["fertile", "defect"]
        ) if prediction_count._metrics else 0,
    }
