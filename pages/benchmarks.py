"""Benchmarks page."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

dash.register_page(__name__, path="/benchmarks")

MARKDOWN_CONTENT = """
### Coming soon.
"""

layout = dbc.Container([html.Div([dcc.Markdown(MARKDOWN_CONTENT)])])
