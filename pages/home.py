"""Home page."""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

dash.register_page(__name__, path="/")

readme = Path(__file__).parent.absolute() / ".." / "README.md"

with open(readme, encoding="utf-8") as f:
    MARKDOWN_CONTENT = f.read()

layout = dbc.Container([html.Div([dcc.Markdown(MARKDOWN_CONTENT)])])
