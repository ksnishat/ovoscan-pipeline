"""
OvoScan Dash Utilities
Helper functions for the dashboard
"""

import os
import requests
from typing import Dict, Any, Optional
import pandas as pd


API_URL = os.getenv("API_URL", "http://localhost:8000")


def call_api(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None, timeout: int = 30) -> Optional[Dict]:
    """
    Generic API call helper

    Args:
        endpoint: API endpoint path
        method: HTTP method
        data: JSON data for POST/PUT
        files: Files for multipart upload
        timeout: Request timeout in seconds

    Returns:
        Response JSON or None on error
    """
    url = f"{API_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data, timeout=timeout)
            else:
                response = requests.post(url, json=data, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}", "detail": response.text}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API"}
    except Exception as e:
        return {"error": str(e)}


def format_confidence(confidence: float) -> str:
    """Format confidence score as percentage"""
    return f"{confidence:.1%}"


def format_processing_time(ms: float) -> str:
    """Format processing time in milliseconds"""
    if ms < 1000:
        return f"{ms:.1f} ms"
    else:
        return f"{ms/1000:.2f} s"


def get_defect_color(class_name: str) -> str:
    """Get color for defect class"""
    colors = {
        "crack": "danger",
        "scratch": "warning",
        "dent": "info",
        "corrosion": "dark",
        "contamination": "secondary",
        "missing_part": "primary",
    }
    return colors.get(class_name.lower(), "muted")


def create_metric_card(title: str, value: str, color: str = "primary", icon: str = None) -> Any:
    """Create a Bootstrap metric card"""
    import dash_bootstrap_components as dbc
    from dash import html

    content = [
        html.H6(title, className="card-title text-muted"),
        html.H3(value, className=f"card-text text-{color}"),
    ]
    if icon:
        content.insert(0, html.I(className=f"bi {icon} fa-2x text-{color} mb-2"))

    return dbc.Card(
        dbc.CardBody(content),
        className="shadow-sm h-100"
    )


def empty_figure(title: str = "No Data") -> Any:
    """Create an empty Plotly figure with a message"""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(
        text=title,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=30, b=20)
    )
    return fig