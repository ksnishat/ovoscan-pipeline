"""
OvoScan Plotly Dash Dashboard
Multi-page enterprise dashboard for visual inspection quality control
"""

import os
import sys
from pathlib import Path

import dash
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import requests
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.dash_app.layouts import create_navbar, create_home_layout, create_inspect_layout, create_metrics_layout, create_models_layout
from src.dash_app.callbacks import register_callbacks

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
DASH_PORT = int(os.getenv("DASH_PORT", "8050"))
DASH_DEBUG = os.getenv("DASH_DEBUG", "false").lower() == "true"

# Initialize Dash app with Bootstrap theme
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="OvoScan - Visual Inspection Dashboard",
    update_title="Loading...",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

server = app.server

# App layout with routing
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    create_navbar(),
    html.Div(id="page-content", className="container-fluid mt-4"),
    dcc.Store(id="api-url-store", data=API_URL),
    dcc.Interval(id="health-check-interval", interval=30000, n_intervals=0),
    html.Div(id="health-status", style={"display": "none"})
])

# Register callbacks
register_callbacks(app)

# Health check endpoint for Kubernetes
@server.route("/_dash-health")
def health_check():
    return {"status": "healthy", "service": "ovoscan-dash"}, 200

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=DASH_PORT,
        debug=DASH_DEBUG
    )