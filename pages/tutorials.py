"""Home page."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc

dash.register_page(__name__, path="/tutorials", order=5)


DATADIR = Path(__file__).absolute().parent / ".." / "assets"

NOTEBOOKS = [os.path.basename(n) for n in glob.glob(str(DATADIR / "*.md"))]


@callback(
    Output("notebook-markdown", "children"),
    Input("notebook-dropdown", "value"),
)
def display_notebook(nb):
    """
    Display the selected notebook by updating the iframe's source.

    This callback updates the src attribute of an iframe component based
    on the value selected from a dropdown menu. It dynamically generates
    the path to the desired notebook file located in the assets directory
    and sets it as the src for rendering in the iframe.

    Parameters:
    nb : str
        The value selected from the dropdown menu, representing the
        notebook filename.

    Returns:
    str
        The dynamically constructed path to the selected notebook file
        located under the 'assets' directory.
    """
    with open(DATADIR / nb) as f:
        return f.read()


HEADER = """
#### Tutorials

We have created a series of Jupyter Notebook tutorials on how to use MatPES. This page provides an easy way to explore
the tutorials.  The Jupyter notebooks can be downloaded from the
[MatPES Github repository](https://github.com/materialsvirtuallab/matpes/tree/main/notebooks).
"""


layout = dbc.Container(
    [
        dcc.Markdown(HEADER),
        dcc.Dropdown(
            id="notebook-dropdown",
            placeholder="Select a notebook to view:",
            value=NOTEBOOKS[0],
            options=[{"label": f.rsplit(".")[0], "value": f} for f in NOTEBOOKS],
        ),
        dcc.Markdown(id="notebook-markdown", style={"marginTop": "10px"}),
    ]
)
