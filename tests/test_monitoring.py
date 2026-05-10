"""Tests for the model monitoring utilities."""
import pytest
from src.utils.monitoring import (
    PredictionTracker,
    record_prediction,
    record_inference_error,
    set_model_loaded,
    get_monitoring_summary,
    tracker,
)


class TestPredictionTracker:
    """Test the PredictionTracker class."""

    def test_tracker_initial_state(self):
        """Tracker should start empty."""
        t = PredictionTracker(window_size=10)
        assert t.get_class_distribution() == {}
        assert t.get_avg_confidence() == 0.0
        assert t.get_prediction_rate() == 0.0

    def test_tracker_records_predictions(self):
        """Tracker should record predictions correctly."""
        t = PredictionTracker(window_size=10)
        t.record("fertile", 0.95)
        t.record("defect", 0.87)
        t.record("fertile", 0.92)

        dist = t.get_class_distribution()
        assert dist["fertile"] == pytest.approx(2 / 3, abs=0.01)
        assert dist["defect"] == pytest.approx(1 / 3, abs=0.01)

    def test_tracker_window_size(self):
        """Tracker should respect window size."""
        t = PredictionTracker(window_size=3)
        for i in range(10):
            t.record("fertile", 0.9)

        assert len(t.predictions) == 3

    def test_tracker_avg_confidence(self):
        """Tracker should compute average confidence."""
        t = PredictionTracker(window_size=10)
        t.record("fertile", 0.8)
        t.record("fertile", 0.9)
        t.record("fertile", 0.7)

        assert t.get_avg_confidence() == pytest.approx(0.8, abs=0.01)

    def test_tracker_reset(self):
        """Tracker should reset correctly."""
        t = PredictionTracker(window_size=10)
        t.record("fertile", 0.9)
        t.reset()
        assert t.get_class_distribution() == {}
        assert t.get_avg_confidence() == 0.0


class TestMonitoringMetrics:
    """Test Prometheus metric recording functions."""

    def test_record_prediction(self):
        """Should record prediction without error."""
        record_prediction("fertile", 0.95, 0.05)
        # If we get here without exception, the test passes
        # (Prometheus metrics are global and don't need assertion)

    def test_record_inference_error(self):
        """Should record inference error without error."""
        record_inference_error()

    def test_set_model_loaded(self):
        """Should set model loaded gauge."""
        set_model_loaded(True)
        set_model_loaded(False)

    def test_get_monitoring_summary(self):
        """Should return a valid monitoring summary."""
        summary = get_monitoring_summary()
        assert "prediction_rate_per_sec" in summary
        assert "avg_confidence" in summary
        assert "class_distribution" in summary
        assert "window_size" in summary
        assert "total_predictions" in summary
        assert isinstance(summary["prediction_rate_per_sec"], float)
        assert isinstance(summary["avg_confidence"], float)
        assert isinstance(summary["class_distribution"], dict)
        assert isinstance(summary["window_size"], int)
