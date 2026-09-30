"""
OvoScan Dash Callbacks
All callback registrations for the multi-page dashboard
"""

import os
import base64
import io
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Input, Output, State, callback_context, no_update, html, dcc, ALL
import dash_bootstrap_components as dbc
from PIL import Image
import numpy as np


API_URL = os.getenv("API_URL", "http://localhost:8000")


def register_callbacks(app):
    """Register all callbacks for the Dash app"""

    # Page routing
    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname")
    )
    def render_page(pathname):
        from src.dash_app.layouts import (
            create_home_layout, create_inspect_layout,
            create_metrics_layout, create_models_layout
        )
        if pathname == "/inspect":
            return create_inspect_layout()
        elif pathname == "/metrics":
            return create_metrics_layout()
        elif pathname == "/models":
            return create_models_layout()
        else:
            return create_home_layout()

    # API status badge
    @app.callback(
        Output("api-status-badge", "children"),
        Output("api-status-badge", "color"),
        Input("health-check-interval", "n_intervals")
    )
    def update_api_status(n):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                if data.get("model_loaded"):
                    return "API: Connected ✓", "success"
                return "API: Model Not Loaded", "warning"
            return "API: Error", "danger"
        except Exception:
            return "API: Disconnected", "danger"

    # Home page KPIs
    @app.callback(
        Output("kpi-total-inspections", "children"),
        Output("kpi-defect-rate", "children"),
        Output("kpi-avg-confidence", "children"),
        Output("kpi-active-models", "children"),
        Input("health-check-interval", "n_intervals")
    )
    def update_home_kpis(n):
        try:
            response = requests.get(f"{API_URL}/stats", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return (
                    f"{data.get('total_inspections', 0):,}",
                    f"{data.get('defect_rate', 0):.1%}",
                    f"{data.get('avg_confidence', 0):.1%}",
                    str(data.get('active_models', 0))
                )
        except Exception:
            pass
        return "—", "—", "—", "—"

    # Activity chart
    @app.callback(
        Output("activity-chart", "figure"),
        Input("health-check-interval", "n_intervals")
    )
    def update_activity_chart(n):
        try:
            response = requests.get(f"{API_URL}/stats/activity", timeout=5)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                fig = px.line(df, x="timestamp", y="count", title="Inspections Over Time")
                fig.update_layout(template="plotly_white", height=300, margin=dict(t=30, b=20))
                return fig
        except Exception:
            pass
        return go.Figure().update_layout(template="plotly_white", height=300)

    # Defect distribution
    @app.callback(
        Output("defect-distribution-chart", "figure"),
        Input("health-check-interval", "n_intervals")
    )
    def update_defect_distribution(n):
        try:
            response = requests.get(f"{API_URL}/stats/defects", timeout=5)
            if response.status_code == 200:
                data = response.json()
                fig = px.pie(values=data.get("values", []), names=data.get("labels", []), title="Defect Types")
                fig.update_layout(template="plotly_white", height=300, margin=dict(t=30, b=20))
                return fig
        except Exception:
            pass
        return go.Figure().update_layout(template="plotly_white", height=300)

    # Performance trend
    @app.callback(
        Output("performance-trend-chart", "figure"),
        Input("health-check-interval", "n_intervals")
    )
    def update_performance_trend(n):
        try:
            response = requests.get(f"{API_URL}/stats/performance", timeout=5)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                fig = go.Figure()
                for metric in ["map50", "map5095", "precision", "recall"]:
                    if metric in df.columns:
                        fig.add_trace(go.Scatter(x=df["date"], y=df[metric], name=metric, mode="lines+markers"))
                fig.update_layout(template="plotly_white", height=350, margin=dict(t=30, b=20))
                return fig
        except Exception:
            pass
        return go.Figure().update_layout(template="plotly_white", height=350)

    # Image upload preview
    @app.callback(
        Output("upload-preview", "children"),
        Output("run-inspection-btn", "disabled"),
        Input("upload-image", "contents"),
        State("upload-image", "filename")
    )
    def update_upload_preview(contents, filename):
        if contents is None:
            return None, True

        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        try:
            image = Image.open(io.BytesIO(decoded))
            # Resize for preview
            image.thumbnail((400, 300))
            buffered = io.BytesIO()
            image.save(buffered, format=image.format or "PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return html.Img(src=f"data:image/{image.format.lower()};base64,{img_str}", style={"maxWidth": "100%"}), False
        except Exception as e:
            return html.Div(f"Error loading image: {e}", className="text-danger"), True

    # Run inspection
    @app.callback(
        Output("inspection-results", "children"),
        Output("detection-image-container", "children"),
        Output("detection-details-table", "children"),
        Output("quality-report", "children"),
        Output("download-report-btn", "disabled"),
        Input("run-inspection-btn", "n_clicks"),
        State("upload-image", "contents"),
        State("upload-image", "filename"),
        prevent_initial_call=True
    )
    def run_inspection(n_clicks, contents, filename):
        if not contents:
            return no_update, no_update, no_update, no_update, no_update

        content_type, content_string = contents.split(",")
        files = {"file": (filename, base64.b64decode(content_string), content_type)}

        try:
            # Call detection endpoint
            response = requests.post(f"{API_URL}/detect", files=files, timeout=30)
            if response.status_code != 200:
                return html.Div(f"Detection failed: {response.text}", className="text-danger"), None, None, None, True

            result = response.json()

            # Create results display
            results_card = dbc.Alert([
                html.H5("Inspection Complete", className="alert-heading"),
                html.P(f"Image: {filename}"),
                html.P(f"Detections: {len(result.get('detections', []))}"),
                html.P(f"Processing Time: {result.get('processing_time_ms', 0):.1f} ms"),
            ], color="success")

            # Display image with detections
            if "annotated_image" in result:
                img_data = result["annotated_image"]
                img_html = html.Img(src=f"data:image/png;base64,{img_data}", style={"maxWidth": "100%"})
            else:
                img_html = html.Div("Annotated image not available", className="text-muted")

            # Detection details table
            detections = result.get("detections", [])
            if detections:
                df = pd.DataFrame(detections)
                table = dbc.Table.from_dataframe(
                    df[["class_name", "confidence", "bbox"]].round({"confidence": 3}),
                    striped=True, bordered=True, hover=True, responsive=True
                )
            else:
                table = html.P("No defects detected", className="text-muted")

            # Quality report
            if "report" in result:
                report = dbc.Card([
                    dbc.CardBody([
                        html.H6("AI Quality Assessment"),
                        html.P(result["report"], className="mb-0"),
                    ])
                ])
            else:
                report = html.Div("Report generation failed or not available", className="text-muted")

            return results_card, img_html, table, report, False

        except Exception as e:
            return html.Div(f"Error: {str(e)}", className="text-danger"), None, None, None, True

    # Metrics page callbacks
    @app.callback(
        Output("metric-map50", "children"),
        Output("metric-map5095", "children"),
        Output("metric-precision", "children"),
        Output("metric-recall", "children"),
        Input("model-selector", "value")
    )
    def update_model_metrics(model_name):
        try:
            response = requests.get(f"{API_URL}/models/{model_name}/metrics", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return (
                    f"{data.get('map50', 0):.3f}",
                    f"{data.get('map5095', 0):.3f}",
                    f"{data.get('precision', 0):.3f}",
                    f"{data.get('recall', 0):.3f}"
                )
        except Exception:
            pass
        return "—", "—", "—", "—"

    @app.callback(
        Output("confusion-matrix", "figure"),
        Input("model-selector", "value")
    )
    def update_confusion_matrix(model_name):
        try:
            response = requests.get(f"{API_URL}/models/{model_name}/confusion-matrix", timeout=5)
            if response.status_code == 200:
                data = response.json()
                fig = px.imshow(data["matrix"], labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=data["labels"], y=data["labels"], text_auto=True)
                fig.update_layout(template="plotly_white", height=300)
                return fig
        except Exception:
            pass
        return go.Figure().update_layout(template="plotly_white", height=300)

    @app.callback(
        Output("pr-curve", "figure"),
        Input("model-selector", "value")
    )
    def update_pr_curve(model_name):
        try:
            response = requests.get(f"{API_URL}/models/{model_name}/pr-curve", timeout=5)
            if response.status_code == 200:
                data = response.json()
                fig = go.Figure()
                for i, (precision, recall) in enumerate(zip(data["precision"], data["recall"])):
                    fig.add_trace(go.Scatter(x=recall, y=precision, name=data["labels"][i], mode="lines"))
                fig.update_layout(template="plotly_white", height=300, xaxis_title="Recall", yaxis_title="Precision")
                return fig
        except Exception:
            pass
        return go.Figure().update_layout(template="plotly_white", height=300)

    @app.callback(
        Output("latency-distribution", "figure"),
        Input("health-check-interval", "n_intervals")
    )
    def update_latency_distribution(n):
        try:
            response = requests.get(f"{API_URL}/stats/latency", timeout=5)
            if response.status_code == 200:
                data = response.json()
                fig = px.histogram(x=data, nbins=30, title="Inference Latency (ms)")
                fig.update_layout(template="plotly_white", height=300)
                return fig
        except Exception:
            pass
        return go.Figure().update_layout(template="plotly_white", height=300)

    @app.callback(
        Output("confidence-distribution", "figure"),
        Input("health-check-interval", "n_intervals")
    )
    def update_confidence_distribution(n):
        try:
            response = requests.get(f"{API_URL}/stats/confidence", timeout=5)
            if response.status_code == 200:
                data = response.json()
                fig = px.histogram(x=data, nbins=30, title="Confidence Scores")
                fig.update_layout(template="plotly_white", height=300)
                return fig
        except Exception:
            pass
        return go.Figure().update_layout(template="plotly_white", height=300)

    # Models page callbacks
    @app.callback(
        Output("models-table", "children"),
        Input("health-check-interval", "n_intervals")
    )
    def update_models_table(n):
        try:
            response = requests.get(f"{API_URL}/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if not data:
                    return html.P("No models registered", className="text-muted")

                df = pd.DataFrame(data)
                return dbc.Table.from_dataframe(
                    df[["name", "version", "stage", "created_at", "metrics.map50"]],
                    striped=True, bordered=True, hover=True, responsive=True
                )
        except Exception:
            pass
        return html.P("Unable to load models", className="text-danger")

    @app.callback(
        Output("compare-model-1", "options"),
        Output("compare-model-2", "options"),
        Input("health-check-interval", "n_intervals")
    )
    def update_model_dropdowns(n):
        try:
            response = requests.get(f"{API_URL}/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                options = [{"label": f"{m['name']} v{m['version']}", "value": m['name']} for m in data]
                return options, options
        except Exception:
            pass
        return [], []

    @app.callback(
        Output("model-comparison-chart", "figure"),
        Input("compare-model-1", "value"),
        Input("compare-model-2", "value")
    )
    def update_model_comparison(model1, model2):
        if not model1 or not model2:
            return go.Figure().update_layout(template="plotly_white", height=300)

        try:
            response = requests.post(f"{API_URL}/models/compare", json={"models": [model1, model2]}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                fig = go.Figure()
                categories = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"]
                fig.add_trace(go.Bar(name=model1, x=categories, y=data[model1]))
                fig.add_trace(go.Bar(name=model2, x=categories, y=data[model2]))
                fig.update_layout(barmode="group", template="plotly_white", height=300)
                return fig
        except Exception:
            pass
        return go.Figure().update_layout(template="plotly_white", height=300)