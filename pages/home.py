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

MARKDOWN_CONTENT = "\n".join(MARKDOWN_CONTENT.split("\n")[2:])

layout = dbc.Container(
    [
        html.H1("MatPES", id="matpes-title"),
        html.H2("A Foundational Potential Energy Surface Dataset for Materials", id="matpes-tagline"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "PBE",
                        href="https://mavrl-web.s3.us-east-1.amazonaws.com/matpes/MatPES-PBE-20240214.json.gz",
                        class_name="ms-2 download-button btn-lg",
                        color="success",
                        external_link=True,
                    ),
                    width={"size": 2, "offset": 4},
                    align="center",
                ),
                dbc.Col(
                    dbc.Button(
                        "r2SCAN",
                        href="https://mavrl-web.s3.us-east-1.amazonaws.com/matpes/MatPES-r2SCAN-20240214.json.gz",
                        class_name="ms-2 download-button btn-lg",
                        color="info",
                        external_link=True,
                    ),
                    width=2,
                    align="center",
                ),
            ],
        ),
        dbc.Row(
            html.Div([dcc.Markdown(MARKDOWN_CONTENT)]),
        ),
    ]
)
