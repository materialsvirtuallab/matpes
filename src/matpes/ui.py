"""Main MatPES Explorer App."""

from __future__ import annotations

import collections
import functools
import itertools
import json
from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc
import plotly.express as px
from dash import Dash, Input, Output, State, callback, dcc, html
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme
from pymatgen.core import Element

from matpes.db import MatPESDB
from matpes.utils import pt_heatmap

if TYPE_CHECKING:
    import pandas as pd

# Define constants
FUNCTIONALS = ("PBE", "r2SCAN")

DB = MatPESDB()


@functools.lru_cache
def get_full_data(functional: str) -> pd.DataFrame:
    """Cache data for each functional for more responsive UI."""
    return DB.get_df(functional)


def get_data(
    functional: str,
    element_filter: list,
    chemsys_filter: str,
    min_coh_e_filter,
    max_coh_e_filter,
    min_form_e_filter,
    max_form_e_filter,
) -> pd.DataFrame:
    """
    Filter data based on the selected functional, element, and chemical system.

    Args:
        functional (str): Functional to filter data for.
        element_filter (list | None): Elements to filter (if any).
        chemsys_filter (str | None): Chemical system to filter (if any).
        min_coh_e_filter (float): Minimum cohesive energy filter.
        max_coh_e_filter (float): Maximum cohesive energy filter.
        min_form_e_filter (float): Minimum form energy filter.
        max_form_e_filter (float): Maximum form energy filter.

    Returns:
        pd.DataFrame: Filtered data.
    """
    df = get_full_data(functional)
    if element_filter:
        df = df[df["elements"].apply(lambda x: set(x).issuperset(element_filter))]
    if chemsys_filter:
        sorted_chemsys = "-".join(sorted(chemsys_filter.split("-")))
        df = df[df["chemsys"] == sorted_chemsys]

    df = df[min_coh_e_filter <= df["cohesive_energy_per_atom"]]
    df = df[df["cohesive_energy_per_atom"] <= max_coh_e_filter]
    df = df[min_form_e_filter <= df["formation_energy_per_atom"]]
    return df[df["formation_energy_per_atom"] <= max_form_e_filter]


@callback(
    [
        Output("min_coh_e_filter", "value"),
        Output("max_coh_e_filter", "value"),
        Output("min_form_e_filter", "value"),
        Output("max_form_e_filter", "value"),
    ],
    [
        Input("functional", "value"),
    ],
)
def update_sliders(functional):
    """Update sliders based on functional."""
    df = get_full_data(functional)
    coh_energy = df["cohesive_energy_per_atom"]
    form_energy = df["formation_energy_per_atom"]
    return coh_energy.min(), coh_energy.max(), form_energy.min(), form_energy.max()


@callback(
    [
        Output("ptheatmap", "figure"),
        Output("coh_energy_hist", "figure"),
        Output("form_energy_hist", "figure"),
        Output("natoms_hist", "figure"),
        Output("nelements_hist", "figure"),
        Output("data-table", "columns"),
        Output("data-table", "data"),
    ],
    [
        Input("functional", "value"),
        Input("el_filter", "value"),
        Input("chemsys_filter", "value"),
        Input("min_coh_e_filter", "value"),
        Input("max_coh_e_filter", "value"),
        Input("min_form_e_filter", "value"),
        Input("max_form_e_filter", "value"),
    ],
)
def update_graph(
    functional, el_filter, chemsys_filter, min_coh_e_filter, max_coh_e_filter, min_form_e_filter, max_form_e_filter
):
    """Update graphs based on user inputs."""
    df = get_data(
        functional, el_filter, chemsys_filter, min_coh_e_filter, max_coh_e_filter, min_form_e_filter, max_form_e_filter
    )
    element_counts = collections.Counter(itertools.chain(*df["elements"]))
    heatmap_figure = pt_heatmap(element_counts, label="Count", log=True)

    table_df = df.drop("elements", axis=1)
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
        [
            {
                "name": i,
                "id": i,
                "type": "numeric",
                "format": Format(precision=4, scheme=Scheme.fixed),
            }
            if i in ["energy", "cohesive_energy_per_atom", "formation_energy_per_atom"]
            else {
                "name": i,
                "id": i,
            }
            for i in table_df.columns
        ],
        table_df.to_dict("records"),
    )


# Callback to download data
@callback(
    Output("download-json", "data"),
    Input("json-download", "n_clicks"),
    State("functional", "value"),
    State("el_filter", "value"),
    State("chemsys_filter", "value"),
    State("min_coh_e_filter", "value"),
    State("max_coh_e_filter", "value"),
    State("min_form_e_filter", "value"),
    State("max_form_e_filter", "value"),
    prevent_initial_call=True,
)
def download_json(
    n_clicks,
    functional,
    el_filter,
    chemsys_filter,
    min_coh_e_filter,
    max_coh_e_filter,
    min_form_e_filter,
    max_form_e_filter,
):
    """Handle json download requests."""
    criteria = {}
    if el_filter:
        criteria["elements"] = el_filter
    if chemsys_filter:
        criteria["chemsys"] = "-".join(sorted(chemsys_filter.split("-")))
    criteria["cohesive_energy_per_atom"] = {"$gte": min_coh_e_filter, "$lte": max_coh_e_filter}
    criteria["formation_energy_per_atom"] = {"$gte": min_form_e_filter, "$lte": max_form_e_filter}
    data = DB.get_json(functional, criteria)
    for entry in data:
        entry.pop("_id", None)  # Remove MongoDB's internal ID
    return dict(
        content=json.dumps(data), filename=f"matpes_{functional}_{el_filter or 'all'}_{chemsys_filter or 'all'}.json"
    )


@callback(
    Output("download-csv", "data"),
    Input("csv-download", "n_clicks"),
    State("functional", "value"),
    State("el_filter", "value"),
    State("chemsys_filter", "value"),
    State("min_coh_e_filter", "value"),
    State("max_coh_e_filter", "value"),
    State("min_form_e_filter", "value"),
    State("max_form_e_filter", "value"),
    prevent_initial_call=True,
)
def download_csv(
    n_clicks,
    functional,
    el_filter,
    chemsys_filter,
    min_coh_e_filter,
    max_coh_e_filter,
    min_form_e_filter,
    max_form_e_filter,
):
    """Handle csv download requests."""
    df = get_data(
        functional, el_filter, chemsys_filter, min_coh_e_filter, max_coh_e_filter, min_form_e_filter, max_form_e_filter
    )
    return dict(content=df.to_csv(), filename=f"matpes_{functional}_{el_filter or 'all'}_{chemsys_filter or 'all'}.csv")


@callback(Output("el_filter", "value"), Input("ptheatmap", "clickData"), State("el_filter", "value"))
def display_click_data(clickdata, el_filter):
    """
    Update el filter when PT table is clicked.

    Args:
        clickdata (dict): Click data.
        el_filter (dict): Element filter.
    """
    new_el_filter = el_filter or []
    if clickdata:
        new_el_filter = {*new_el_filter, Element.from_Z(clickdata["points"][0]["pointNumber"] + 1).symbol}
    return list(new_el_filter)


def main():
    """Main entry point for MatPES Explorer UI."""
    app = Dash("MatPES Explorer", external_stylesheets=[dbc.themes.CERULEAN], title="MatPES Explorer")

    # Define app layout
    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                html.Img(
                                    src="https://github.com/materialsvirtuallab/matpes/blob"
                                    "/2b7f8de716289de8089504a63c6431c456268172/assets/logo.png?raw=true",
                                    width="50%",
                                    style={
                                        "padding": "12px",
                                    },
                                ),
                                className="text-primary text-center",
                            )
                        ],
                        width={"size": 6, "offset": 3},
                    ),
                ]
            ),
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
                    )
                ]
            ),
            dbc.Row(
                [
                    html.Div("Filters: "),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Element(s)"),
                            dcc.Dropdown(
                                id="el_filter",
                                options=[
                                    {"label": el.symbol, "value": el.symbol}
                                    for el in Element
                                    if el.name not in ("D", "T")
                                ],
                                multi=True,
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            html.Div("Chemsys"),
                            dcc.Input(
                                id="chemsys_filter",
                                placeholder="Li-Fe-O",
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            html.Div("Coh. Energy (Min, Max)"),
                            dcc.Input(0, type="number", id="min_coh_e_filter"),
                            dcc.Input(10, type="number", id="max_coh_e_filter"),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("Form. Energy (Min, Max)"),
                            dcc.Input(0, type="number", id="min_form_e_filter"),
                            dcc.Input(10, type="number", id="max_form_e_filter"),
                        ],
                        width=4,
                    ),
                ]
            ),
            dbc.Row(
                [
                    html.Div("Download"),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Button("JSON", id="json-download"),
                            dcc.Download(id="download-json"),
                        ],
                        width=1,
                    ),
                    dbc.Col(
                        [
                            html.Button("CSV", id="csv-download"),
                            dcc.Download(id="download-csv"),
                        ],
                        width=1,
                    ),
                ]
            ),
            html.Div(
                [
                    html.P("Help:"),
                    html.Ul(
                        [
                            html.Li("Clicking on the PT adds an element to the element filter."),
                            html.Li(
                                "Element filter is restrictive, i.e., only data containing all selected elements + any "
                                "other elements are shown."
                            ),
                            html.Li(
                                "Chemsys filter: Only data within the chemsys are shown. Typically you should only"
                                " use either element or chemsys but not both."
                            ),
                        ]
                    ),
                ],
                style={"padding": 5},
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [dcc.Graph(id="ptheatmap", figure=pt_heatmap({}, label=""))],
                        style={"marginLeft": "auto", "marginRight": "auto", "text-align": "center"},
                    ),
                    width=12,
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
            dbc.Row([DataTable(page_size=25, id="data-table")]),
        ]
    )

    app.run(debug=True)
