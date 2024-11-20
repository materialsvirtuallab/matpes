"""Main MatPES Explorer App."""

from __future__ import annotations

import collections
import itertools
import json
from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc
import plotly.express as px
from dash import Dash, Input, Output, State, callback, dcc, html
from pymatgen.core import Element

from matpes.db import MatPESDB
from matpes.utils import pt_heatmap

if TYPE_CHECKING:
    import pandas as pd

# Define constants
FUNCTIONALS = ("PBE", "r2SCAN")

DB = MatPESDB()


def get_data(functional: str, element_filter: list, chemsys: str) -> pd.DataFrame:
    """
    Filter data based on the selected functional, element, and chemical system.

    Args:
        functional (str): Functional to filter data for.
        element_filter (list | None): Elements to filter (if any).
        chemsys (str | None): Chemical system to filter (if any).

    Returns:
        pd.DataFrame: Filtered data.
    """
    df = DB.get_df(functional)
    if element_filter:
        df = df[df["elements"].apply(lambda x: set(x).issuperset(element_filter))]
    if chemsys:
        sorted_chemsys = "-".join(sorted(chemsys.split("-")))
        df = df[df["chemsys"] == sorted_chemsys]
    return df


@callback(
    [
        Output("ptheatmap", "figure"),
        Output("coh_energy_hist", "figure"),
        Output("form_energy_hist", "figure"),
        Output("natoms_hist", "figure"),
        Output("nelements_hist", "figure"),
    ],
    [Input("functional", "value"), Input("el_filter", "value"), Input("chemsys_filter", "value")],
)
def update_graph(functional, el_filter, chemsys_filter):
    """Update graphs based on user inputs."""
    df = get_data(functional, el_filter, chemsys_filter)
    element_counts = collections.Counter(itertools.chain(*df["elements"]))
    heatmap_figure = pt_heatmap(element_counts, label="Count", log=True)
    return (
        heatmap_figure,
        px.histogram(
            df,
            x="cohesive_energy_per_atom",
            labels={"cohesive_energy_per_atom": "Cohesive Energy per Atom (eV/atom)"},
            nbins=100,
        ),
        px.histogram(
            df,
            x="formation_energy_per_atom",
            labels={"formation_energy_per_atom": "Formation Energy per Atom (eV/atom)"},
            nbins=100,
        ),
        px.histogram(df, x="natoms"),
        px.histogram(df, x="nelements"),
    )


# Callback to download data
@callback(
    Output("download-data", "data"),
    Input("btn-download", "n_clicks"),
    State("functional", "value"),
    State("el_filter", "value"),
    State("chemsys_filter", "value"),
    prevent_initial_call=True,
)
def download_data(n_clicks, functional, el_filter, chemsys_filter):
    """Handle data download requests."""
    collection = DB[functional]
    criteria = {}
    if el_filter:
        criteria["elements"] = el_filter
    if chemsys_filter:
        criteria["chemsys"] = "-".join(sorted(chemsys_filter.split("-")))
    data = list(collection.find(criteria))
    for entry in data:
        entry.pop("_id", None)  # Remove MongoDB's internal ID
    return dict(
        content=json.dumps(data), filename=f"matpes_{functional}_{el_filter or 'all'}_{chemsys_filter or 'all'}.json"
    )


def main():
    """Main entry point for MatPES Explorer UI."""
    app = Dash("MatPES Explorer", external_stylesheets=[dbc.themes.CERULEAN], title="MatPES Explorer")

    # Define app layout
    app.layout = dbc.Container(
        [
            dbc.Row([html.Div("MatPES Explorer", className="text-primary text-center fs-3")]),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Functional"),
                            dcc.Dropdown(
                                id="functional",
                                options=[{"label": f, "value": f} for f in FUNCTIONALS],
                                value="PBE",
                                clearable=False,
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            html.Label("Filter by Element(s)"),
                            dcc.Dropdown(
                                id="el_filter",
                                options=[{"label": el.symbol, "value": el.symbol} for el in Element],
                                multi=True,
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            html.Label("Filter by Chemsys"),
                            dcc.Input(
                                id="chemsys_filter",
                                placeholder="Li-Fe-O",
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            html.Label("Data Tools"),
                            html.Button("Download", id="btn-download"),
                            dcc.Download(id="download-data"),
                        ],
                        width=1,
                    ),
                ]
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(id="ptheatmap", style={"marginLeft": "auto", "marginRight": "auto"}),
                    width={"size": 8, "offset": 2},
                )
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="coh_energy_hist"), width=6),
                    dbc.Col(dcc.Graph(id="form_energy_hist"), width=6),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="natoms_hist"), width=6),
                    dbc.Col(dcc.Graph(id="nelements_hist"), width=6),
                ]
            ),
        ]
    )

    app.run(debug=True)
