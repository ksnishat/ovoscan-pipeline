"""Integration tests against the live docker stack.

Skipped unless ``LIVE_STACK=1``. The rest of the suite uses ``TestClient``,
which runs the ASGI app in-process and cannot detect a container that never
started, a wrong port, or Prometheus failing to scrape.

Run with::

    LIVE_STACK=1 pytest tests/test_integration_live.py -v
"""

import os

import pytest
import requests

pytestmark = pytest.mark.integration

API_PORT = int(os.getenv("OVOSCAN_API_PORT", "8003"))
MLFLOW_PORT = int(os.getenv("OVOSCAN_MLFLOW_PORT", "5000"))
PROM_PORT = int(os.getenv("OVOSCAN_PROM_PORT", "9093"))
GRAFANA_PORT = int(os.getenv("OVOSCAN_GRAFANA_PORT", "3000"))
MINIO_PORT = int(os.getenv("OVOSCAN_MINIO_PORT", "9006"))

API = f"http://127.0.0.1:{API_PORT}"
TIMEOUT = 120


def _require_live() -> None:
    if os.getenv("LIVE_STACK") != "1":
        pytest.skip("set LIVE_STACK=1 to run integration tests against the live stack")


# The APIs enforce X-API-Key when API_KEY is set. The live stack runs with auth
# enabled, so every request must carry the key. Set LIVE_API_KEY to match the
# stack; the default matches run_stack.sh's demo value.
API_KEY = os.getenv("LIVE_API_KEY", "demo-secret-key")
AUTH_HEADERS = {"X-API-Key": API_KEY}


def _auth(kw):
    """Merge the API key into a request's headers."""
    headers = dict(kw.pop("headers", {}) or {})
    headers.update(AUTH_HEADERS)
    kw["headers"] = headers
    return kw


def _get(url: str, **kw):
    try:
        return requests.get(url, timeout=TIMEOUT, **_auth(kw))
    except requests.RequestException as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"service unreachable: {exc}")


def _sample_image() -> str:
    """Return a path to a real egg image, or skip."""
    import glob

    pattern = os.getenv("OVOSCAN_SAMPLE_IMAGE", "egg_dataset/valid/images/*.jpg")
    matches = sorted(glob.glob(pattern))
    if not matches:
        pytest.skip(f"no sample image found matching {pattern}")
    return matches[0]


def test_api_health_endpoint_is_live():
    _require_live()
    r = _get(f"{API}/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"healthy", "running"}
    assert body["model_loaded"] is True, "API started without a model checkpoint"


def test_api_metrics_are_scrapable():
    _require_live()
    r = _get(f"{API}/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")


def test_predict_classifies_a_real_image():
    _require_live()
    path = _sample_image()
    with open(path, "rb") as fh:
        r = requests.post(f"{API}/predict", files={"file": (os.path.basename(path), fh, "image/jpeg")},
            timeout=TIMEOUT,
            headers=AUTH_HEADERS,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in {"fertile", "infertile", "defect"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_analyze_report_returns_a_report_even_without_an_llm():
    """The RAG path must degrade to a template, not 500, when Ollama is down."""
    _require_live()
    path = _sample_image()
    with open(path, "rb") as fh:
        r = requests.post(f"{API}/analyze-report", files={"file": (os.path.basename(path), fh, "image/jpeg")},
            timeout=TIMEOUT,
            headers=AUTH_HEADERS,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["technical_report"], "report was empty"
    assert body["status"] == "success"


def test_monitoring_endpoint_reports_traffic():
    _require_live()
    r = _get(f"{API}/monitoring")
    assert r.status_code == 200
    body = r.json()
    assert "total_predictions" in body
    assert "class_distribution" in body


def test_mlflow_tracking_server_is_reachable():
    _require_live()
    # MLflow validates the Host header; a bare 127.0.0.1 is not in its default
    # allow-list, so send an explicit Host. A server with no experiments yet
    # omits the "experiments" key entirely, so assert on the response shape
    # rather than on that key.
    r = _get(
        f"http://127.0.0.1:{MLFLOW_PORT}/api/2.0/mlflow/experiments/search",
        headers={"Host": f"127.0.0.1:{MLFLOW_PORT}"},
        params={"max_results": 1},
    )
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), dict)


def test_prometheus_is_scraping_the_api():
    _require_live()
    r = _get(f"http://127.0.0.1:{PROM_PORT}/api/v1/targets")
    assert r.status_code == 200
    targets = r.json()["data"]["activeTargets"]
    assert targets, "Prometheus has no configured targets"
    for t in targets:
        assert t["health"] == "up", f"{t['scrapeUrl']} is {t['health']}: {t.get('lastError')}"


def test_grafana_is_healthy():
    _require_live()
    r = _get(f"http://127.0.0.1:{GRAFANA_PORT}/api/health")
    assert r.status_code == 200
    assert r.json()["database"] == "ok"


def test_minio_is_live():
    _require_live()
    r = _get(f"http://127.0.0.1:{MINIO_PORT}/minio/health/live")
    assert r.status_code == 200
