"""Tests for PSI drift detection.

The important properties are: identical distributions give PSI 0, a real shift
gives a large PSI, degenerate inputs do not raise, and the monitor only reports
once it has enough samples to be meaningful.
"""

import json

import numpy as np
import pytest

from src.utils.drift import (
    DriftMonitor,
    classify_psi,
    compute_psi,
)


class TestComputePsi:
    def test_identical_distributions_give_zero(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(0, 1, 5000)
        assert compute_psi(sample, sample) == pytest.approx(0.0, abs=1e-9)

    def test_small_shift_is_stable(self):
        rng = np.random.default_rng(1)
        base = rng.normal(0, 1, 5000)
        shifted = rng.normal(0.1, 1, 5000)
        psi = compute_psi(base, shifted)
        assert psi < 0.10, f"a 0.1-sigma shift should be stable, got {psi}"

    def test_large_shift_is_significant(self):
        rng = np.random.default_rng(2)
        base = rng.normal(0, 1, 5000)
        shifted = rng.normal(1.0, 1, 5000)
        psi = compute_psi(base, shifted)
        assert psi >= 0.25, f"a 1-sigma shift should be significant, got {psi}"

    def test_psi_is_non_negative(self):
        rng = np.random.default_rng(3)
        base = rng.normal(0, 1, 2000)
        for _ in range(5):
            assert compute_psi(base, rng.normal(0, 1, 2000)) >= 0.0

    def test_constant_feature_does_not_raise(self):
        """A sensor that never varies collapses the quantile bins."""
        assert compute_psi(np.ones(100), np.ones(100)) == 0.0

    def test_empty_input_returns_zero(self):
        assert compute_psi([], np.arange(100.0)) == 0.0
        assert compute_psi(np.arange(100.0), []) == 0.0

    def test_nan_values_are_ignored(self):
        base = np.concatenate([np.arange(100.0), [np.nan, np.inf]])
        assert compute_psi(base, np.arange(100.0)) == pytest.approx(0.0, abs=1e-9)


class TestClassifyPsi:
    @pytest.mark.parametrize(
        "psi,expected",
        [
            (0.0, "stable"),
            (0.09, "stable"),
            (0.10, "moderate"),
            (0.24, "moderate"),
            (0.25, "significant"),
            (5.0, "significant"),
        ],
    )
    def test_thresholds(self, psi, expected):
        assert classify_psi(psi) == expected

    def test_non_finite_is_unknown(self):
        assert classify_psi(float("nan")) == "unknown"


class TestDriftMonitor:
    def _baseline(self):
        rng = np.random.default_rng(7)
        return {"s1": rng.normal(0, 1, 1000).tolist()}

    def test_disabled_without_baseline(self):
        monitor = DriftMonitor(baseline={})
        assert monitor.enabled is False
        assert monitor.compute() == {}

    def test_enabled_with_baseline(self):
        assert DriftMonitor(baseline=self._baseline()).enabled is True

    def test_requires_minimum_samples(self):
        """PSI on a handful of points is noise, so it must not be reported."""
        monitor = DriftMonitor(baseline=self._baseline())
        for _ in range(5):
            monitor.observe({"s1": 0.0})
        assert monitor.compute(min_samples=30) == {}

    def test_reports_after_enough_samples(self):
        monitor = DriftMonitor(baseline=self._baseline())
        rng = np.random.default_rng(8)
        for value in rng.normal(0, 1, 200):
            monitor.observe({"s1": float(value)})
        results = monitor.compute(min_samples=30)
        assert "s1" in results
        assert results["s1"]["status"] == "stable"

    def test_detects_a_real_shift(self):
        monitor = DriftMonitor(baseline=self._baseline())
        rng = np.random.default_rng(9)
        for value in rng.normal(3.0, 1, 300):
            monitor.observe({"s1": float(value)})
        results = monitor.compute(min_samples=30)
        assert results["s1"]["status"] == "significant"
        assert monitor.max_psi(min_samples=30) >= 0.25

    def test_unknown_features_are_ignored(self):
        monitor = DriftMonitor(baseline=self._baseline())
        monitor.observe({"not_a_sensor": 1.0})
        assert monitor.compute(min_samples=1) == {}

    def test_non_numeric_values_are_ignored(self):
        monitor = DriftMonitor(baseline=self._baseline())
        monitor.observe({"s1": "not a number"})
        assert monitor.compute(min_samples=1) == {}

    def test_window_is_bounded(self):
        monitor = DriftMonitor(baseline=self._baseline(), window_size=10)
        for i in range(100):
            monitor.observe({"s1": float(i)})
        assert len(monitor._observed["s1"]) == 10

    def test_reset_keeps_baseline(self):
        monitor = DriftMonitor(baseline=self._baseline())
        monitor.observe({"s1": 1.0})
        monitor.reset()
        assert monitor.enabled is True
        assert monitor.compute(min_samples=1) == {}

    def test_round_trip_through_file(self, tmp_path):
        path = tmp_path / "baseline.json"
        monitor = DriftMonitor(baseline=self._baseline())
        monitor.save_baseline(str(path))

        reloaded = DriftMonitor.from_file(str(path))
        assert reloaded.enabled is True
        assert set(reloaded.baseline) == {"s1"}

    def test_missing_file_yields_disabled_monitor(self, tmp_path):
        monitor = DriftMonitor.from_file(str(tmp_path / "absent.json"))
        assert monitor.enabled is False

    def test_saved_baseline_is_valid_json(self, tmp_path):
        path = tmp_path / "baseline.json"
        DriftMonitor(baseline=self._baseline()).save_baseline(str(path))
        payload = json.loads(path.read_text())
        assert "features" in payload
        assert "s1" in payload["features"]


class TestOvoScanPredictionDrift:
    """OvoScan compares the live prediction distribution to a baseline.

    A counter alone cannot detect drift: it has no reference to compare against.
    These tests pin that PSI is actually computed and interpreted.
    """

    def _reset(self):
        from src.utils import monitoring

        monitoring.tracker.reset()
        monitoring.set_drift_baseline({"fertile": 0.5, "infertile": 0.5})

    def test_balanced_stream_is_stable(self):
        from src.utils import monitoring

        self._reset()
        for i in range(60):
            monitoring.record_prediction("fertile" if i % 2 == 0 else "infertile", 0.9, 0.01)
        assert monitoring.compute_drift_psi() == pytest.approx(0.0, abs=1e-6)
        assert monitoring.get_monitoring_summary()["drift_status"] == "stable"

    def test_skewed_stream_is_significant(self):
        from src.utils import monitoring

        self._reset()
        for i in range(100):
            monitoring.record_prediction("fertile" if i % 20 else "infertile", 0.9, 0.01)
        psi = monitoring.compute_drift_psi()
        assert psi is not None and psi >= 0.25
        assert monitoring.get_monitoring_summary()["drift_status"] == "significant"

    def test_insufficient_samples_returns_none(self):
        from src.utils import monitoring

        self._reset()
        for _ in range(5):
            monitoring.record_prediction("fertile", 0.9, 0.01)
        assert monitoring.compute_drift_psi() is None
        assert monitoring.get_monitoring_summary()["drift_status"] == "insufficient_data"

    def test_baseline_is_normalised(self):
        from src.utils import monitoring

        monitoring.set_drift_baseline({"fertile": 80, "infertile": 20})
        baseline = monitoring.get_drift_baseline()
        assert baseline["fertile"] == pytest.approx(0.8)
        assert baseline["infertile"] == pytest.approx(0.2)

    def test_zero_total_baseline_is_ignored(self):
        from src.utils import monitoring

        monitoring.set_drift_baseline({"fertile": 0.5, "infertile": 0.5})
        monitoring.set_drift_baseline({"fertile": 0, "infertile": 0})
        # An all-zero baseline is meaningless, so the previous one is kept.
        assert monitoring.get_drift_baseline()["fertile"] == pytest.approx(0.5)

    def test_disappearing_class_still_counts(self):
        """A class that vanishes entirely must register as drift."""
        from src.utils import monitoring

        self._reset()
        for _ in range(100):
            monitoring.record_prediction("fertile", 0.9, 0.01)
        psi = monitoring.compute_drift_psi()
        assert psi is not None and psi >= 0.25


class TestBaselineMatchesModelClasses:
    """The drift baseline must use the class names the model actually emits.

    The classifier reports {0: 'defect', 1: 'fertile'}. A baseline keyed on
    "infertile" compares against a class that never appears, so PSI reports a
    large value for every stream — including a perfectly balanced one — which
    makes the metric useless.
    """

    def test_default_baseline_uses_model_class_names(self):
        from src.utils import monitoring

        baseline = monitoring.DEFAULT_BASELINE
        assert set(baseline) == {"fertile", "defect"}, (
            f"baseline keys {set(baseline)} do not match the model's classes "
            "{'fertile', 'defect'}"
        )

    def test_balanced_stream_against_default_baseline_is_stable(self):
        from src.utils import monitoring

        monitoring.tracker.reset()
        monitoring.set_drift_baseline(monitoring.DEFAULT_BASELINE)
        for i in range(60):
            monitoring.record_prediction("fertile" if i % 2 == 0 else "defect", 0.9, 0.01)
        psi = monitoring.compute_drift_psi()
        assert psi is not None
        assert psi < 0.10, f"a balanced stream should be stable, got PSI {psi}"

    def test_skewed_stream_against_default_baseline_is_significant(self):
        from src.utils import monitoring

        monitoring.tracker.reset()
        monitoring.set_drift_baseline(monitoring.DEFAULT_BASELINE)
        for _ in range(100):
            monitoring.record_prediction("defect", 0.9, 0.01)
        psi = monitoring.compute_drift_psi()
        assert psi is not None and psi >= 0.25


class TestOllamaModelResolution:
    """A configured tag that is not pulled must not fail silently.

    The API previously used the code default "llama3.2" while only "llama3.2:1b"
    was pulled, so /analyze-report fell back to a template while /health still
    reported rag_available: true.
    """

    TAGS = ["llama3.2:1b", "llava:7b", "gemma:latest"]

    def test_present_tag_is_used_as_is(self):
        from src.agent.rag import resolve_llm_model

        model, warning = resolve_llm_model("llama3.2:1b", "http://x", self.TAGS)
        assert model == "llama3.2:1b"
        assert warning is None

    def test_missing_tag_falls_back_to_the_same_family(self):
        from src.agent.rag import resolve_llm_model

        model, warning = resolve_llm_model("llama3.2", "http://x", self.TAGS)
        assert model == "llama3.2:1b"
        assert warning and "llama3.2" in warning

    def test_unreachable_server_keeps_the_requested_tag(self):
        from src.agent.rag import resolve_llm_model

        model, warning = resolve_llm_model("llama3.2", "http://x", [])
        assert model == "llama3.2"
        assert warning is None

    def test_warning_names_the_available_tags(self):
        from src.agent.rag import resolve_llm_model

        _, warning = resolve_llm_model("llama3.2", "http://x", self.TAGS)
        for tag in self.TAGS:
            assert tag in warning

    def test_hatchery_agent_actually_calls_the_resolver(self, monkeypatch):
        """The resolver existing is not enough; it must be wired in."""
        from src.agent import rag

        calls = []

        def spy(requested, base_url, available=None):
            calls.append(requested)
            return "spy-model", "spy-warning"

        monkeypatch.setattr(rag, "resolve_llm_model", spy)
        monkeypatch.setattr(rag, "HuggingFaceEmbeddings", lambda **kw: object())
        monkeypatch.setattr(rag, "Ollama", lambda **kw: object())

        agent = rag.HatcheryAgent(knowledge_base_path="x", llm_model="llama3.2")

        assert calls == ["llama3.2"], "HatcheryAgent did not call resolve_llm_model"
        assert agent.llm_model_name == "spy-model"
        assert agent.model_warning == "spy-warning"
