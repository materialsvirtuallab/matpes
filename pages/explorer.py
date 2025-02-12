"""Main MatPES Explorer App."""

from __future__ import annotations

import collections
import itertools
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, callback, dcc, html
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme
from pymatgen.core import Element

from .utils import pt_heatmap

dash.register_page(__name__)

# Define constants
FUNCTIONALS = ("PBE", "r2SCAN")

DATADIR = Path(__file__).absolute().parent


def get_data(
    functional: str,
    el_filter: list,
    chemsys_filter: str,
    min_coh_e_filter,
    max_coh_e_filter,
    min_form_e_filter,
    max_form_e_filter,
) -> pd.DataFrame:
    """
    Filter data.

    Args:
        functional (str): The functional used to filter the dataset (e.g., "PBE", "r2SCAN").
        el_filter (list of str): A list of element symbols to filter the dataset by (e.g., ["Fe", "Ni"]).
        chemsys_filter (list of str): A list of chemical systems to filter by (e.g., ["Fe-O", "Ni-Mn"]).
        min_coh_e_filter (float): Minimum cohesive energy per atom to include in the dataset.
        max_coh_e_filter (float): Maximum cohesive energy per atom to include in the dataset.
        min_form_e_filter (float): Minimum formation energy per atom to include in the dataset.
        max_form_e_filter (float): Maximum formation energy per atom to include in the dataset.

    Returns:
        pd.DataFrame: Filtered data.
    """
    df = pd.read_pickle(DATADIR / f"{functional.lower()}_stats.pkl")
    if el_filter:
        df = df[df["elements"].apply(lambda x: set(x).issuperset(el_filter))]
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
    """Update sliders based on functional.

    Args:
        functional (str): The functional used to filter the dataset (e.g., "PBE", "LDA").
    """
    df = pd.read_pickle(DATADIR / f"{functional.lower()}_stats.pkl")
    coh_energy = df["cohesive_energy_per_atom"]
    form_energy = df["formation_energy_per_atom"]
    return coh_energy.min(), coh_energy.max(), form_energy.min(), form_energy.max()


@callback(
    [
        Output("pt-div", "children"),
        Output("stats-div", "children"),
        Output("data-div", "children"),
    ],
    [
        Input("functional", "value"),
        Input("el_filter", "value"),
        Input("chemsys_filter", "value"),
        Input("min_coh_e_filter", "value"),
        Input("max_coh_e_filter", "value"),
        Input("min_form_e_filter", "value"),
        Input("max_form_e_filter", "value"),
        Input("display_options", "value"),
    ],
)
def display_data(
    functional,
    el_filter,
    chemsys_filter,
    min_coh_e_filter,
    max_coh_e_filter,
    min_form_e_filter,
    max_form_e_filter,
    display_options,
):
    """
    Update graphs and data tables based on user-provided filters and criteria.

    This function processes the input filters and generates various visualizations and data structures,
    including a heatmap, histograms, and a formatted data table. The data is derived from a dataset
    filtered by the specified parameters.

    Args:
        functional (str): The functional used to filter the dataset (e.g., "PBE", "r2SCAN").
        el_filter (list of str): A list of element symbols to filter the dataset by (e.g., ["Fe", "Ni"]).
        chemsys_filter (list of str): A list of chemical systems to filter by (e.g., ["Fe-O", "Ni-Mn"]).
        min_coh_e_filter (float): Minimum cohesive energy per atom to include in the dataset.
        max_coh_e_filter (float): Maximum cohesive energy per atom to include in the dataset.
        min_form_e_filter (float): Minimum formation energy per atom to include in the dataset.
        max_form_e_filter (float): Maximum formation energy per atom to include in the dataset.
        display_options (list of str): A list of display options.

    Returns:
        tuple:
            - heatmap_figure (plotly.graph_objects.Figure): A heatmap of element counts, displayed in log scale.
            - histograms of formation energies, cohesive energies, nsites, nlements.
            - data table.
    """
    df = get_data(
        functional, el_filter, chemsys_filter, min_coh_e_filter, max_coh_e_filter, min_form_e_filter, max_form_e_filter
    )
    element_counts = collections.Counter(itertools.chain(*df["elements"]))
    nstructures = len(df)
    output = [
        dbc.Row(
            [
                html.H4(f"Elemental Heatmap of {nstructures:,} Structures in Dataset", className="section-title"),
                dbc.Col(
                    html.Div(
                        [dcc.Graph(id="ptheatmap", figure=pt_heatmap(element_counts, label="Count", log=True))],
                        style={"marginLeft": "auto", "marginRight": "auto", "text-align": "center"},
                    ),
                    width=12,
                ),
            ]
        )
    ]
    table_df = df.drop("elements", axis=1)
    output.append(
        dbc.Row(
            [
                html.H4("Property distribution of dataset", className="section-title"),
                dbc.Col(
                    dcc.Graph(
                        id="coh_energy_hist",
                        figure=px.histogram(
                            df,
                            x="cohesive_energy_per_atom",
                            labels={"cohesive_energy_per_atom": "Cohesive Energy per Atom (eV/atom)"},
                            nbins=30,
                        ),
                    ),
                    width=6,
                ),
                dbc.Col(
                    dcc.Graph(
                        id="form_energy_hist",
                        figure=px.histogram(
                            df,
                            x="formation_energy_per_atom",
                            labels={"formation_energy_per_atom": "Formation Energy per Atom (eV/atom)"},
                            nbins=30,
                        ),
                    ),
                    width=6,
                ),
                dbc.Col(dcc.Graph(id="nsites_hist", figure=px.histogram(df, x="nsites")), width=6),
                dbc.Col(
                    dcc.Graph(id="nelements_hist", figure=px.histogram(df, x="nelements")),
                    width=6,
                ),
            ]
        )
    )
    if display_options and "Show Table" in display_options:
        output.append(
            DataTable(
                page_size=25,
                id="data-table",
                columns=[
                    {
                        "name": i,
                        "id": i,
                        "type": "numeric",
                        "format": Format(precision=3, scheme=Scheme.fixed),
                    }
                    if i in ["energy", "cohesive_energy_per_atom", "formation_energy_per_atom"]
                    else {
                        "name": i,
                        "id": i,
                    }
                    for i in table_df.columns
                ],
                data=table_df.to_dict("records"),
            )
        )
    else:
        output.append("")
    return output


@callback(Output("el_filter", "value"), Input("ptheatmap", "clickData"), State("el_filter", "value"))
def update_el_filter_on_click(clickdata, el_filter):
    """
    Update el filter when PT table is clicked.

    Args:
        clickdata (dict): Click data.
        el_filter (dict): Element filter.
    """
    new_el_filter = el_filter or []
    if clickdata:
        try:
            z = clickdata["points"][0]["text"].split("<")[0]
            new_el_filter = {*new_el_filter, Element.from_Z(int(z)).symbol}
        except (ValueError, AttributeError):
            pass
    return list(new_el_filter)


# Define app layout
layout = dbc.Container(
    [
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
                                {"label": el.symbol, "value": el.symbol} for el in Element if el.name not in ("D", "T")
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
                html.Div("Options (note: enabling these will slow rendering)"),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Checklist(options=["Show Table"], id="display_options"),
                    ],
                    width=4,
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
        html.Div([html.H1("Loading...")], id="pt-div"),
        html.Div(id="stats-div"),
        html.Div(id="data-div"),
        # dbc.Row([DataTable(page_size=25, id="data-table")]),
    ]
)
