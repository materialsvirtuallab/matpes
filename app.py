"""Main MatPES Explorer App."""
from __future__ import annotations

import collections
import functools
import itertools

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import pymatviz as pmv
from dash import Dash, Input, Output, callback, dcc, html
from pymatgen.core import Element
from pymongo import MongoClient

FUNCTIONALS = ("PBE", "r2SCAN")

# Set up MongoDB client and database
CLIENT = MongoClient()
DB = CLIENT["matpes"]
#print(DB["PBE"].find_one())
RAW_DATA = {}
for f in FUNCTIONALS:
    collection = DB[f]
    RAW_DATA[f] = pd.DataFrame(
        collection.find({}, projection=["elements", "energy", "cohesive_energy_per_atom", "formation_energy", "natoms"])
        )

@functools.lru_cache
def get_data(functional, el):
    """Filter data with caching for improved performance."""
    df = RAW_DATA[functional]
    if el is not None:
        df = df[df["elements"].apply(lambda x: el in x)]
    return df

# Initialize the Dash app with a Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div("MatPES Explorer", className="text-primary text-center fs-3")
    ]),
    dbc.Row([dcc.RadioItems(
        options=[
            {"label": f, "value": f} for f in FUNCTIONALS
        ],
        value="PBE",
        id="functional"
    ),
    dcc.Dropdown(
        options=[
            {"label": el.symbol, "value": el.symbol} for el in Element
        ],
        id="el_filter"
    )]),
    dbc.Row([
    dcc.Graph(
        id="ptheatmap"
    )]),
    dbc.Row([
    dcc.Graph(
        id="coh_energy_hist"
    ),
    dcc.Graph(
        id="form_energy_hist"
    )
    ])
])

# Define callback to update the heatmap based on selected functional
@callback(
    [Output("ptheatmap", "figure"), Output("coh_energy_hist", "figure"), Output("form_energy_hist", "figure")],
    [Input("functional", "value"), Input("el_filter", "value")]
)
def update_graph(functional, el_filter):
    """Update graph based on input."""
    df = get_data(functional, el_filter)
    el_count = collections.Counter(itertools.chain(*df["elements"]))
    heatmap_figure = pmv.ptable_heatmap_plotly(el_count, log=True)
    return heatmap_figure, px.histogram(df, x="cohesive_energy_per_atom"), px.histogram(df, x="formation_energy")


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
