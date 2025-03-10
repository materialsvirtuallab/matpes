"""Home page."""

from __future__ import annotations

from pathlib import Path
import glob
import os

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback

dash.register_page(__name__, path="/tutorials", order=5)


DATADIR = Path(__file__).absolute().parent / ".." / "assets"

NOTEBOOKS = [os.path.basename(n) for n in glob.glob(str(DATADIR / "*.html"))]


@callback(
    Output("tutorial-iframe", "src"),
    Input("notebook-dropdown", "value"),
)
def display_notebook(nb):
    if nb is not None:
        return f"assets/{nb}"


layout = dbc.Container(
    [
        dcc.Dropdown(
            id="notebook-dropdown",
            placeholder="Select a notebook to view:",
            value=NOTEBOOKS[0],
            options=[{"label": f, "value": f} for f in NOTEBOOKS],
        ),
        html.Iframe(
            style={"height": "1000px", "width": "100%"},
            id="tutorial-iframe",
        ),
    ]
)
