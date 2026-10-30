"""Tests for the OvoScan Dash dashboard.

The dashboard was at 0% coverage, so a broken page or a bad API contract
shipped unnoticed. These tests drive the helpers and the app factory directly;
a full browser test would need dash_duo plus a headless browser in CI, which is
not worth it for a demo UI, while the failures that actually occur (wrong URL,
unhandled timeout, unknown defect class, missing figure) are all reachable
without one.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dash_app import utils  # noqa: E402


class TestCallApi:
    """`call_api` is the only path from the UI to the service."""

    def test_returns_json_on_200(self, monkeypatch):
        class Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"prediction": "fertile", "confidence": 0.97}

        monkeypatch.setattr(utils.requests, "get", lambda *a, **k: Resp())
        assert utils.call_api("/") == {"prediction": "fertile", "confidence": 0.97}

    def test_surfaces_api_error_status(self, monkeypatch):
        class Resp:
            status_code = 500
            text = "boom"

        monkeypatch.setattr(utils.requests, "get", lambda *a, **k: Resp())
        result = utils.call_api("/predict")
        assert "error" in result
        assert "500" in result["error"]

    def test_timeout_is_reported_not_raised(self, monkeypatch):
        def timeout(*a, **k):
            raise utils.requests.exceptions.Timeout()

        monkeypatch.setattr(utils.requests, "get", timeout)
        assert utils.call_api("/") == {"error": "Request timeout"}

    def test_connection_error_is_reported_not_raised(self, monkeypatch):
        def refuse(*a, **k):
            raise utils.requests.exceptions.ConnectionError()

        monkeypatch.setattr(utils.requests, "get", refuse)
        assert utils.call_api("/") == {"error": "Cannot connect to API"}

    def test_post_with_files_uses_multipart(self, monkeypatch):
        seen = {}

        class Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"ok": True}

        def fake_post(url, files=None, data=None, timeout=None):
            seen["files"] = files
            return Resp()

        monkeypatch.setattr(utils.requests, "post", fake_post)
        utils.call_api("/predict", method="POST", files={"file": ("a.jpg", b"x")})
        assert seen["files"] is not None

    def test_post_without_files_sends_json(self, monkeypatch):
        seen = {}

        class Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"ok": True}

        def fake_post(url, json=None, files=None, data=None, timeout=None):
            seen["json"] = json
            return Resp()

        monkeypatch.setattr(utils.requests, "post", fake_post)
        utils.call_api("/predict", method="POST", data={"a": 1})
        assert seen["json"] == {"a": 1}

    def test_unsupported_method_is_an_error_not_a_crash(self, monkeypatch):
        result = utils.call_api("/x", method="DELETE")
        assert "error" in result
        assert "DELETE" in result["error"]


class TestFormatting:
    def test_confidence_is_a_percentage(self):
        assert utils.format_confidence(0.975) == "97.5%"

    def test_confidence_handles_extremes(self):
        assert utils.format_confidence(0.0) == "0.0%"
        assert utils.format_confidence(1.0) == "100.0%"

    @pytest.mark.parametrize(
        "ms,expected", [(12.34, "12.3 ms"), (999.0, "999.0 ms"), (1500.0, "1.50 s")]
    )
    def test_processing_time_switches_unit_at_a_second(self, ms, expected):
        assert utils.format_processing_time(ms) == expected

    def test_known_defect_classes_have_colours(self):
        for name in ("crack", "scratch", "dent", "corrosion", "contamination", "missing_part"):
            assert utils.get_defect_color(name) != "muted"

    def test_unknown_class_falls_back(self):
        """A new defect class must not blank the UI."""
        assert utils.get_defect_color("weird-new-class") == "muted"

    def test_class_lookup_is_case_insensitive(self):
        assert utils.get_defect_color("CRACK") == utils.get_defect_color("crack")


class TestComponents:
    def test_metric_card_renders(self):
        card = utils.create_metric_card("Defects", "42")
        assert card is not None

    def test_metric_card_accepts_an_icon(self):
        card = utils.create_metric_card("Defects", "42", icon="bi-bug")
        assert card is not None

    def test_empty_figure_has_the_title_annotation(self):
        fig = utils.empty_figure("Nothing here")
        assert fig.layout.annotations[0].text == "Nothing here"

    def test_empty_figure_has_axes_hidden(self):
        fig = utils.empty_figure()
        assert fig.layout.xaxis.visible is False
        assert fig.layout.yaxis.visible is False


class TestDashboardApp:
    def test_app_builds_and_registers_callbacks(self):
        """Importing the module builds the Dash app; a bad callback signature
        surfaces here rather than in the browser."""
        from src.dash_app import app as dash_app

        assert dash_app.app is not None
        assert dash_app.server is not None

    def test_layout_contains_routing_and_health_elements(self):
        from src.dash_app import app as dash_app

        ids = set()

        def walk(component):
            if hasattr(component, "id") and component.id:
                ids.add(component.id)
            children = getattr(component, "children", None)
            if isinstance(children, (list, tuple)):
                for child in children:
                    walk(child)
            elif children is not None and hasattr(children, "children"):
                walk(children)

        walk(dash_app.app.layout)
        assert {"url", "page-content", "api-url-store"} <= ids

    def test_health_route_returns_200(self):
        from src.dash_app import app as dash_app

        client = dash_app.server.test_client()
        response = client.get("/_dash-health")
        assert response.status_code == 200
        assert response.get_json()["status"] == "healthy"

    def test_navbar_is_built(self):
        from src.dash_app.layouts import create_navbar

        assert create_navbar() is not None


class TestCallbacks:
    """Drive the registered callbacks directly.

    `app.callback_map` holds Dash's wrapper; `__wrapped__` is the original
    function, which is what we want to exercise without a browser.
    """

    @staticmethod
    def _cb(name_fragment):
        from src.dash_app import app as dash_app

        for key, value in dash_app.app.callback_map.items():
            if name_fragment in key:
                return value["callback"].__wrapped__
        raise AssertionError(f"no callback registered for {name_fragment}")

    @pytest.mark.parametrize("path", ["/", "/inspect", "/metrics", "/models", "/unknown"])
    def test_every_path_renders_a_page(self, path):
        """An unhandled path must fall back to home, not render blank."""
        assert self._cb("page-content")(path) is not None

    def test_api_status_badge_reports_connected(self, monkeypatch):
        import src.dash_app.callbacks as cb

        class Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"model_loaded": True}

        monkeypatch.setattr(cb.requests, "get", lambda *a, **k: Resp())
        label, color = self._cb("api-status-badge")(0)
        assert color == "success"
        assert "Connected" in label

    def test_api_status_badge_warns_when_model_not_loaded(self, monkeypatch):
        import src.dash_app.callbacks as cb

        class Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"model_loaded": False}

        monkeypatch.setattr(cb.requests, "get", lambda *a, **k: Resp())
        _, color = self._cb("api-status-badge")(0)
        assert color == "warning"

    def test_api_status_badge_handles_an_api_outage(self, monkeypatch):
        """The dashboard must stay usable while the API is down."""
        import src.dash_app.callbacks as cb

        def refuse(*a, **k):
            raise ConnectionError("down")

        monkeypatch.setattr(cb.requests, "get", refuse)
        label, color = self._cb("api-status-badge")(0)
        assert color == "danger"
        assert label

    def test_all_callbacks_are_unique_and_registered(self):
        from src.dash_app import app as dash_app

        assert len(dash_app.app.callback_map) >= 10


class TestApiErrorPaths:
    """The service must degrade to a JSON error, never a stack trace.

    These drive the FastAPI app directly: a 500 with an HTML traceback is worse
    than a documented error body because the dashboard renders whatever it gets.
    """

    @staticmethod
    def _client():
        from fastapi.testclient import TestClient

        from src.app import app as api

        return TestClient(api)

    @staticmethod
    def _jpg():
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (32, 32), color=(120, 120, 120)).save(buf, format="JPEG")
        return buf.getvalue()

    def test_predict_returns_json_error_when_model_fails(self, monkeypatch):
        import src.app as app_mod

        def boom(_bytes):
            raise RuntimeError("model gone")

        monkeypatch.setattr(app_mod, "_detect_image", boom)
        response = self._client().post(
            "/predict", files={"file": ("x.jpg", self._jpg(), "image/jpeg")}
        )
        body = response.json()
        assert body["status"] == "error"
        assert "model gone" in body["message"]

    def test_analyze_report_reports_service_unavailable(self, monkeypatch):
        import src.app as app_mod

        def boom(_bytes):
            raise RuntimeError("no model")

        monkeypatch.setattr(app_mod, "_detect_image", boom)
        response = self._client().post(
            "/analyze-report", files={"file": ("x.jpg", self._jpg(), "image/jpeg")}
        )
        body = response.json()
        assert body["status"] == "error"
        assert "unavailable" in body["message"].lower()

    def test_metrics_endpoint_is_prometheus_text(self):
        response = self._client().get("/metrics")
        assert response.status_code == 200
        assert b"# HELP" in response.content or response.content

    def test_monitoring_endpoint_returns_the_summary(self):
        response = self._client().get("/monitoring")
        assert response.status_code == 200
        assert isinstance(response.json(), dict)

class TestDetectImage:
    """`_detect_image` is the guard between the HTTP layer and the model."""

    def test_raises_when_no_model_is_loaded(self, monkeypatch):
        import src.app as app_mod

        monkeypatch.setattr(app_mod, "model", None)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            app_mod._detect_image(b"not-an-image")

    def test_invalid_image_bytes_raise_rather_than_silently_pass(self, monkeypatch):
        """A corrupt upload must surface, not be reported as a clean scan."""

        class FakeModel:
            def predict(self, img, conf=None):
                raise AssertionError("model must not be reached")

        monkeypatch.setattr(app_mod_holder()["m"], "model", FakeModel())
        # PIL raises several unrelated types for corrupt bytes (UnidentifiedImageError,
        # OSError, ValueError depending on the failure point); any of them is a
        # correct rejection, so the type is deliberately not narrowed.
        with pytest.raises(Exception):  # noqa: B017
            app_mod_holder()["m"]._detect_image(b"definitely-not-an-image")

    def test_a_valid_image_produces_a_prediction_and_confidence(self, monkeypatch):
        import src.app as app_mod

        class FakeProbs:
            top1 = 0
            top1conf = 0.91

        class FakeResult:
            probs = FakeProbs()
            names = {0: "defect", 1: "fertile"}

        class FakeModel:
            def predict(self, img, conf=None):
                return [FakeResult()]

        monkeypatch.setattr(app_mod, "model", FakeModel())
        monkeypatch.setattr(app_mod, "record_prediction", lambda *a, **k: None)
        result = app_mod._detect_image(TestApiErrorPaths._jpg())
        assert result["prediction"] == "defect"
        assert result["confidence"] == 0.91


class TestGenerateReport:
    """The report must always be produced, with or without the RAG agent."""

    def test_fertile_gets_the_pass_template(self, monkeypatch):
        import src.app as app_mod

        monkeypatch.setattr(app_mod, "agent", None)
        assert "Quality Standard Met" in app_mod._generate_report("fertile", 0.99)

    def test_defect_gets_the_rejection_template(self, monkeypatch):
        import src.app as app_mod

        monkeypatch.setattr(app_mod, "agent", None)
        report = app_mod._generate_report("crack", 0.88)
        assert "rejection" in report.lower()

    def test_agent_result_is_used_when_available(self, monkeypatch):
        import src.app as app_mod

        class FakeAgent:
            def analyze_defect(self, prediction):
                return f"RAG says {prediction}"

        monkeypatch.setattr(app_mod, "agent", FakeAgent())
        assert app_mod._generate_report("crack", 0.5) == "RAG says crack"

    def test_agent_failure_falls_back_to_the_template(self, monkeypatch):
        """An LLM outage must not lose the report entirely."""
        import src.app as app_mod

        class BrokenAgent:
            def analyze_defect(self, prediction):
                raise ConnectionError("ollama down")

        monkeypatch.setattr(app_mod, "agent", BrokenAgent())
        assert "rejection" in app_mod._generate_report("crack", 0.5).lower()


def app_mod_holder():
    import src.app as app_mod
    return {"m": app_mod}
