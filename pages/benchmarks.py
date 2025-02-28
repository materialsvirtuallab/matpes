"""Benchmarks page."""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table, dcc, html

dash.register_page(__name__, path="/benchmarks")

DATADIR = Path(__file__).absolute().parent

pbe_df = pd.read_csv(DATADIR / "matcalc-benchmark-pbe.csv")
r2scan_df = pd.read_csv(DATADIR / "matcalc-benchmark-r2scan.csv")

INTRO_CONTENT = """
## Matcalc-Benchmark

The performance of different models trained on different datasets on the Matcalc-Benchmark is shown below. The sizes
of the datasets are:
- MPF: 185,877
- MPtrj: 1,580,395
- OMat24: 100,824,585
- MatPES PBE: 434,712
- MatPES r2SCAN: 387,897
"""

LEGEND = """
Legend:
- Formation energy per atom E_form: meV/atom
- Bulk modulus K_VRH: GPa
- Shear modulus G_VRH: GPa
- Constant volume heat capacity C_V: J/mol/K
- Ionic conductivity sigma: mS/cm
- Median termination temperature T_1/2^term: K
- Time per MD timestep t_step: ms/step/atom
"""


def get_best(df, i):
    """
    Determine the best value from a specified column in a DataFrame.

    This function selects either the maximum or minimum value of a specified column
    based on the input. For specific column names, the maximum value is chosen,
    while for all other columns, the minimum value is selected.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to evaluate.
        i (str): The name of the column to determine the best value from.

    Returns:
        float: The best value from the specified column, determined either as the
        maximum or minimum value.
    """
    if i in ("f_UMLIP/f_DFT", "T_1/2^term (K)"):
        return df[i].max()
    return df[i].min()


layout = dbc.Container(
    [
        dbc.Col(
            html.Div([dcc.Markdown(INTRO_CONTENT)]),
            width=12,
        ),
        dbc.Col(html.H4("PBE"), width=12),
        dbc.Col(
            dash_table.DataTable(
                id="pbe-benchmarks-table",
                columns=[{"name": i.split("(")[0] if "log" not in i else i, "id": i} for i in pbe_df.columns],
                data=pbe_df.to_dict("records"),
                style_data_conditional=[
                    {
                        "if": {
                            "filter_query": f"{{{i}}} = {get_best(pbe_df, i)}",
                            "column_id": i,
                        },
                        "font-weight": "bold",
                    }
                    for i in pbe_df.columns[2:]
                ],
            ),
            width=12,
        ),
        dbc.Col(
            html.H4("r2SCAN"),
            width=12,
            style={"padding-top": "10px"},
        ),
        dbc.Col(
            dash_table.DataTable(
                id="r2scan-benchmarks-table",
                columns=[{"name": i.split("(")[0] if "log" not in i else i, "id": i} for i in r2scan_df.columns],
                data=r2scan_df.to_dict("records"),
                style_data_conditional=[
                    {
                        "if": {
                            "filter_query": f"{{{i}}} = {get_best(r2scan_df, i)}",
                            "column_id": i,
                        },
                        "font-weight": "bold",
                    }
                    for i in r2scan_df.columns[2:]
                ],
            ),
            width=6,
        ),
        dbc.Col(
            html.Div([dcc.Markdown(LEGEND)]),
            width=12,
            style={"padding-top": "10px"},
        ),
    ]
)
