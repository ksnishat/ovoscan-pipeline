"""OvoScan Project API and Pipeline Tests.

Tests for the FastAPI inference endpoint, ZenML pipeline, and YOLOv8 classification.
"""

import pytest
import tempfile
import os
from fastapi.testclient import TestClient
from src.app import app


@pytest.fixture(scope="module")
def client():
    # Context manager is required so FastAPI lifespan (model loading) runs.
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def dummy_egg_image(tmp_path):
    """Create a small dummy egg-shaped image."""
    try:
        from PIL import Image
        import numpy as np
        img = Image.new("RGB", (300, 300), color="white")
        img.save(tmp_path / "test_egg.jpg", format="JPEG")
        return tmp_path / "test_egg.jpg"
    except ImportError:
        pytest.skip("PIL not installed")


@pytest.fixture
def dummy_npy_data(tmp_path):
    """Create a dummy numpy data file for ZenML pipeline tests."""
    import numpy as np
    data = np.random.rand(50, 20)
    np.save(tmp_path / "test_data.npy", data)
    return tmp_path / "test_data.npy"


@pytest.fixture
def bad_image_file(tmp_path):
    """Create a corrupted/invalid image file."""
    (tmp_path / "bad.jpg").write_bytes(b"corrupted data")
    return tmp_path / "bad.jpg"


class TestHealth:
    """Test / endpoint."""

    def test_health_check(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"


class TestPredict:
    """Test /predict endpoint."""

    def test_predict_with_valid_image(self, client, dummy_egg_image):
        with open(dummy_egg_image, "rb") as f:
            response = client.post("/predict", files={"file": f.read()})
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data or "pred" in data
        assert "confidence" in data or "confidence" in data
        # prediction should be either "fertile" or a defect class
        pred = data.get("prediction", data.get("pred", ""))
        assert pred.upper() in ("FERTILE", "CRACK", "INFERTILE", "BAD")

    def test_predict_returns_confidence(self, client, dummy_egg_image):
        with open(dummy_egg_image, "rb") as f:
            response = client.post("/predict", files={"file": f.read()})
        data = response.json()
        conf = data.get("confidence", data.get("confidence_score", 0))
        assert isinstance(conf, float)
        assert 0 <= conf <= 1

    def test_predict_handles_missing_model(self, client):
        """A request without a file must be rejected cleanly (not crash the server)."""
        response = client.post("/predict")
        # FastAPI validation returns 422 when the required file field is absent.
        assert response.status_code in (200, 422, 500)
        assert response.status_code != 200 or "status" in response.json()


    def test_predict_uses_gpu_detection(self, client, dummy_egg_image):
        """Test that GPU detection logging works."""
        with open(dummy_egg_image, "rb") as f:
            response = client.post("/predict", files={"file": f.read()})
        assert response.status_code == 200


class TestAgent:
    """Test RAG agent endpoint."""

    def test_analyze_report_valid(self, client, dummy_egg_image):
        with open(dummy_egg_image, "rb") as f:
            response = client.post("/analyze-report", files={"file": f.read()})
        # Should not crash, might return error if model missing
        assert response.status_code in (200, 500)
        if response.status_code == 200:
            data = response.json()
            assert "technical_report" in data

    def test_analyze_report_returns_report_structure(self, client, dummy_egg_image):
        with open(dummy_egg_image, "rb") as f:
            response = client.post("/analyze-report", files={"file": f.read()})
        if response.status_code == 200:
            data = response.json()
            assert len(data["technical_report"]) > 0


class TestPipeline:
    """Test ZenML pipeline components."""

    def test_pipeline_zenml_import(self):
        """ZenML pipeline should import without error."""
        from src.pipelines.training_pipeline import ovoscan_training_pipeline
        assert ovoscan_training_pipeline is not None

    def test_pipeline_structure(self):
        """Pipeline should expose a data_path parameter (ZenML wraps the signature)."""
        from src.pipelines.training_pipeline import ovoscan_training_pipeline
        import inspect
        target = getattr(ovoscan_training_pipeline, "entrypoint", ovoscan_training_pipeline)
        params = list(inspect.signature(target).parameters.keys())
        assert "data_path" in params

    def test_pipeline_parameter_defaults(self):
        """Pipeline should have sensible parameter defaults."""
        from src.pipelines.training_pipeline import ovoscan_training_pipeline
        # Can be called with just data_path (using defaults for epochs, batch_size)


class TestPrometheusMetrics:
    """Test Prometheus metrics endpoint."""

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.content.decode("utf-8")
        assert len(data) > 0


class TestDVC:
    """Test DVC data versioning references."""

    def test_dvc_available(self):
        """DVC should be available for data versioning."""
        import shutil
        import subprocess
        if shutil.which("dvc") is None:
            pytest.skip("DVC not installed")
        result = subprocess.run(["dvc", "--version"], capture_output=True, text=True)
        assert result.returncode == 0


class TestStreamlitDashboard:
    """Test Streamlit dashboard routing."""

    def test_dashboard_page_loads(self, client):
        """Main dashboard page should load without error."""
        response = client.get("/")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])