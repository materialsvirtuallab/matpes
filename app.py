"""Main MatPES Explorer App."""

from __future__ import annotations

import collections
import functools
import itertools
import json

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import pymatviz as pmv
from dash import Dash, Input, Output, State, callback, dcc, html
from pymatgen.core import Element
from pymongo import MongoClient

FUNCTIONALS = ("PBE", "r2SCAN")

# Set up MongoDB client and database
CLIENT = MongoClient()
DB = CLIENT["matpes"]


@functools.lru_cache
def get_df(functional: str) -> pd.DataFrame:
    collection = DB[functional]
    return pd.DataFrame(
        collection.find(
            {},
            projection=[
                "elements",
                "energy",
                "chemsys",
                "cohesive_energy_per_atom",
                "formation_energy_per_atom",
                "natoms",
                "nelements",
            ],
        )
    )


@functools.lru_cache
def get_data(functional, el, chemsys):
    """Filter data with caching for improved performance."""
    df = get_df(functional)
    if el is not None:
        df = df[df["elements"].apply(lambda x: el in x)]
    if chemsys:
        chemsys = "-".join(sorted(chemsys.split("-")))
        df = df[df["chemsys"] == chemsys]

    return df


# Initialize the Dash app with a Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Define the app layout
app.layout = dbc.Container(
    [
        dbc.Row([html.Div("MatPES Explorer", className="text-primary text-center fs-3")]),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Functional"),
                        dcc.RadioItems(
                            options=[{"label": f, "value": f} for f in FUNCTIONALS], value="PBE", id="functional"
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.Label("Element Filter"),
                        dcc.Dropdown(
                            options=[{"label": el.symbol, "value": el.symbol} for el in Element], id="el_filter"
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.Label("Chemsys Filter"),
                        dcc.Input(id="chemsys_filter", placeholder="Li-Fe-O"),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [html.Div([html.Button("Download", id="btn-download"), dcc.Download(id="download-data")])],
                    width=2,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [dcc.Graph(id="ptheatmap", style={"marginLeft": "auto", "marginRight": "auto"})],
                    width={"size": 8, "order": "last", "offset": 2},
                )
            ],
        ),
        dbc.Row(
            [
                dbc.Col([dcc.Graph(id="coh_energy_hist")], width=6),
                dbc.Col([dcc.Graph(id="form_energy_hist")], width=6),
            ]
        ),
        dbc.Row(
            [
                dbc.Col([dcc.Graph(id="natoms_hist")], width=6),
                dbc.Col([dcc.Graph(id="nelements_hist")], width=6),
            ]
        ),
    ]
)


# Define callback to update the heatmap based on selected functional
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
    """Update graph based on input."""
    df = get_data(functional, el_filter, chemsys_filter)
    el_count = {el.symbol: 0 for el in Element}
    el_count.update(collections.Counter(itertools.chain(*df["elements"])))
    heatmap_figure = pmv.ptable_heatmap_plotly(el_count)
    return (
        heatmap_figure,
        px.histogram(
            df, x="cohesive_energy_per_atom", labels={"cohesive_energy_per_atom": "Cohesive Energy per Atom (eV/atom)"}
        ),
        px.histogram(
            df,
            x="formation_energy_per_atom",
            labels={"formation_energy_per_atom": "Formation Energy per Atom (eV/atom)"},
        ),
        px.histogram(df, x="natoms"),
        px.histogram(df, x="nelements"),
    )


@callback(
    Output("download-data", "data"),
    Input("btn-download", "n_clicks"),
    State("functional", "value"),
    State("el_filter", "value"),
    State("chemsys_filter", "value"),
    prevent_initial_call=True,
)
def download(n_clicks, functional, el_filter, chemsys_filter):
    collection = DB[functional]
    criteria = {}
    if el_filter is not None:
        criteria["elements"] = el_filter
    if chemsys_filter is not None:
        chemsys = "-".join(sorted(chemsys_filter.split("-")))
        criteria["chemsys"] = chemsys
    data = list(collection.find(criteria))
    for d in data:
        del d["_id"]
    return dict(content=json.dumps(data), filename=f"matpes_{functional}_{el_filter}_{chemsys_filter}.json")


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
