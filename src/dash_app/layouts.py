"""
OvoScan Dash Layouts
Page layouts for the multi-page dashboard
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go


def create_navbar():
    """Create the navigation bar"""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/", active="exact")),
            dbc.NavItem(dbc.NavLink("Inspect", href="/inspect", active="exact")),
            dbc.NavItem(dbc.NavLink("Metrics", href="/metrics", active="exact")),
            dbc.NavItem(dbc.NavLink("Models", href="/models", active="exact")),
            dbc.NavItem([
                html.Span(id="api-status-badge", className="badge bg-secondary ms-2"),
            ]),
        ],
        brand=html.Div([
            html.I(className="bi bi-eye me-2"),
            "OvoScan"
        ], className="fw-bold"),
        brand_href="/",
        color="primary",
        dark=True,
        className="mb-4",
        fluid=True,
    )


def create_home_layout():
    """Create the home page layout with KPI cards"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Total Inspections", className="card-title text-muted"),
                        html.H2(id="kpi-total-inspections", className="card-text text-primary"),
                        html.Small("All time", className="text-muted"),
                    ])
                ], className="shadow-sm h-100")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Defect Rate", className="card-title text-muted"),
                        html.H2(id="kpi-defect-rate", className="card-text text-danger"),
                        html.Small("Last 24 hours", className="text-muted"),
                    ])
                ], className="shadow-sm h-100")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Avg Confidence", className="card-title text-muted"),
                        html.H2(id="kpi-avg-confidence", className="card-text text-success"),
                        html.Small("Model confidence", className="text-muted"),
                    ])
                ], className="shadow-sm h-100")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Active Models", className="card-title text-muted"),
                        html.H2(id="kpi-active-models", className="card-text text-info"),
                        html.Small("In production", className="text-muted"),
                    ])
                ], className="shadow-sm h-100")
            ], md=3),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Recent Inspection Activity", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="activity-chart", config={"displayModeBar": False})
                    ])
                ], className="shadow-sm")
            ], md=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Defect Distribution", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="defect-distribution-chart", config={"displayModeBar": False})
                    ])
                ], className="shadow-sm")
            ], md=4),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Model Performance Trends", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="performance-trend-chart", config={"displayModeBar": False})
                    ])
                ], className="shadow-sm")
            ], md=12),
        ])
    ])


def create_inspect_layout():
    """Create the inspection page layout"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Upload Image for Inspection", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Upload(
                            id="upload-image",
                            children=html.Div([
                                html.I(className="bi bi-cloud-upload fa-3x text-muted mb-3"),
                                html.Br(),
                                "Drag and Drop or ",
                                html.A("Select Image"),
                                html.Br(),
                                html.Small("Supported: JPG, PNG, TIFF (max 10MB)", className="text-muted")
                            ]),
                            style={
                                "width": "100%",
                                "height": "200px",
                                "lineHeight": "200px",
                                "borderWidth": "2px",
                                "borderStyle": "dashed",
                                "borderRadius": "10px",
                                "textAlign": "center",
                                "margin": "10px 0",
                            },
                            multiple=False,
                            accept=".jpg,.jpeg,.png,.tiff,.tif"
                        ),
                        html.Div(id="upload-preview"),
                        html.Br(),
                        dbc.Button(
                            [html.I(className="bi bi-search me-2"), "Run Inspection"],
                            id="run-inspection-btn",
                            color="primary",
                            size="lg",
                            className="w-100",
                            disabled=True
                        ),
                    ])
                ], className="shadow-sm mb-4")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Inspection Results", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.Div(id="inspection-results"),
                        html.Div(id="detection-image-container"),
                    ])
                ], className="shadow-sm mb-4")
            ], md=6),
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Detection Details", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.Div(id="detection-details-table")
                    ])
                ], className="shadow-sm")
            ], md=12),
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Quality Assessment Report", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.Div(id="quality-report"),
                        html.Br(),
                        dbc.Button(
                            [html.I(className="bi bi-download me-2"), "Download Report (PDF)"],
                            id="download-report-btn",
                            color="secondary",
                            disabled=True
                        ),
                    ])
                ], className="shadow-sm")
            ], md=12),
        ], className="mt-4")
    ])


def create_metrics_layout():
    """Create the metrics page layout"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Model Metrics", className="mb-0"),
                        dbc.Select(
                            id="model-selector",
                            options=[
                                {"label": "YOLOv8n", "value": "yolov8n"},
                                {"label": "YOLOv8s", "value": "yolov8s"},
                                {"label": "YOLOv8m", "value": "yolov8m"},
                            ],
                            value="yolov8s",
                            style={"width": "200px"}
                        )
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("mAP@0.5", className="card-title"),
                                        html.H3(id="metric-map50", className="text-primary"),
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("mAP@0.5:0.95", className="card-title"),
                                        html.H3(id="metric-map5095", className="text-info"),
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Precision", className="card-title"),
                                        html.H3(id="metric-precision", className="text-success"),
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Recall", className="card-title"),
                                        html.H3(id="metric-recall", className="text-warning"),
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=3),
                        ], className="mb-3"),

                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Confusion Matrix"),
                                    dbc.CardBody([
                                        dcc.Graph(id="confusion-matrix", config={"displayModeBar": False})
                                    ])
                                ], className="shadow-sm")
                            ], md=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("PR Curve"),
                                    dbc.CardBody([
                                        dcc.Graph(id="pr-curve", config={"displayModeBar": False})
                                    ])
                                ], className="shadow-sm")
                            ], md=6),
                        ])
                    ])
                ], className="shadow-sm")
            ], md=12),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Inference Latency Distribution"),
                    dbc.CardBody([
                        dcc.Graph(id="latency-distribution", config={"displayModeBar": False})
                    ])
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Confidence Score Distribution"),
                    dbc.CardBody([
                        dcc.Graph(id="confidence-distribution", config={"displayModeBar": False})
                    ])
                ], className="shadow-sm")
            ], md=6),
        ])
    ])


def create_models_layout():
    """Create the models management page layout"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Registered Models", className="mb-0"),
                        dbc.Button(
                            [html.I(className="bi bi-plus-circle me-2"), "Register New Model"],
                            id="register-model-btn",
                            color="primary",
                            size="sm"
                        )
                    ]),
                    dbc.CardBody([
                        html.Div(id="models-table"),
                        html.Div(id="register-model-modal", className="mt-3")
                    ])
                ], className="shadow-sm")
            ], md=12),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Comparison"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Select(
                                    id="compare-model-1",
                                    placeholder="Select Model 1",
                                    style={"width": "100%"}
                                )
                            ], md=6),
                            dbc.Col([
                                dbc.Select(
                                    id="compare-model-2",
                                    placeholder="Select Model 2",
                                    style={"width": "100%"}
                                )
                            ], md=6),
                        ], className="mb-3"),
                        dcc.Graph(id="model-comparison-chart", config={"displayModeBar": False})
                    ])
                ], className="shadow-sm")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Artifacts"),
                    dbc.CardBody([
                        html.Div(id="model-artifacts-list")
                    ])
                ], className="shadow-sm")
            ], md=6),
        ])
    ])