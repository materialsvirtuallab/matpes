"""Benchmarks page."""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme

dash.register_page(__name__, path="/benchmarks")

DATADIR = Path(__file__).absolute().parent

pbe_df = pd.read_csv(DATADIR / "matcalc-benchmark-pbe.csv")
r2scan_df = pd.read_csv(DATADIR / "matcalc-benchmark-r2scan.csv")

INTRO_CONTENT = """
## MatCalc-Benchmark

The MatCalc-Benchmark is designed to evaluate the performance of different universal machine learning interatomic
potentials (UMLIPs) on a balanced set of equilibrium, near-equilibrium and molecular dynamics properties of materials.

In addition to the property metrics, it is important to consider the training data size and the number of
parameters of the UMLIPs. Large datasets are difficult to train with, requiring large amounts of CPU/GPU resources.
For instance, the training time for TensorNet on the MatPES-PBE dataset is about 15 minutes per epoch on a single
Nvidia RTX A6000 GPU, while that for the same model on the OMat24 dataset is around 20 hours per epoch on eight Nvidia
A100 GPUs. UMLIPs with large number of parameters will be expensive to run in MD simulations (see t_step metric below),
limiting the size of the simulation cells or time scales you can study. For reference, the t_step of eqV2-OMat24 is
around 213 ms/atom/step (~2 orders of magnitude more expensive than the models shown here).

For the initial launch, we have included M3GNet, CHGNet and TensorNet UMLIPs trained on the MatPES, MPF,
MPtrj, and OMat24 datasets. The sizes of the datasets are:
- MPF: 185,877
- MPtrj: 1,580,395
- OMat24: 100,824,585
- MatPES PBE: 434,712
- MatPES r2SCAN: 387,897

We welcome the contribution of other UMLIPs to be added to this MatCalc-Benchmark. To ensure a fair comparison, we
will require all UMLIPs included in the benchmark to provide **information about training dataset size, training cost,
and the number of parameters**, in addition to the performance metrics below. The easiest way to run the benchmark is to
implement an ASE compatible calculator, which can then be used with the
[MatCalc](https://github.com/materialsvirtuallab/matcalc) package. We will release the equilibrium and
near-equilibrium benchmark datasets soon in the MatCalc repository together with benchmarking tools.
"""

TABLE_NOTE = """
For MAEs, all values not statistically different from the best value in each column are highlighted. Statistical
significance is determined using a [paired t-test](https://en.wikipedia.org/wiki/Paired_difference_test) with 0.05
alpha level.
"""

LEGEND = r"""
MatCalc-Benchmark metrics can be divided into three categories: equilibrium, near-equilibrium, and molecular dynamics
properties.

| Task                          | Symbol     | Units        | Functional   | Test Source              | Number |
|-------------------------------|------------|--------------|--------------|--------------------------|--------|
| **Equilibrium**               |            |              |              |                          |        |
| Structural similarity         | \|v\|      | -            | PBE          | [WBM]                    | 1, 000 |
|                               |            | -            | RRSCAN       | [GNoME]                  | 1,000  |
| Formation energy per atom     | Ef         | eV/atom      | PBE          | [WBM]                    | 1,000  |
|                               |            | eV/atom      | RRSCAN       | [GNoME]                  | 1,000  |
| **Near-equilibrium**          |            |              |              |                          |        |
| Bulk modulus                  | K_VRH      | GPa          | PBE          | [MP]                     | 3,959  |
| Shear modulus                 | G_VRH      | GPa          | PBE          | [MP]                     | 3,959  |
| Constant volume heat capacity | C_V        | J/mol/K      | PBE          | [Alexandria]             | 1,170  |
| Off-equilibrium force         | F/F_DFT    | --           | PBE          | [WBM high energy states] | 979    |
| **Molecular dynamics**        |            |              |              |                          |        |
| Median termination temp       | T_1/2^term | K            | PBE & RRSCAN | [MVL]                    | 172    |
| Ionic conductivity            | sigma      | mS/cm        | PBE          | [MVL]                    | 698    |
| Time per atom per time step   | t_step     | ms/step/atom | PBE & RRSCAN | [MVL]                    | 1      |

The time per atom per time step (t_step) was computed using MD simulations conducted on a single Intel Xeon Gold core
for a system of 64 Si atoms under ambient conditions (300 K and 1 bar) over 50 ps with a 1 fs time step.

[WBM]: https://doi.org/10.1038/s41524-020-00481-6
[GNoME]: https://doi.org/10.1038/s41586-023-06735-9
[Alexandria]: https://doi.org/10.48550/arXiv.2412.16551
[WBM high energy states]: https://doi.org/10.48550/arXiv.2405.07105
[MP]: http://materialsproject.org
[MVL]: http://materialsvirtuallab.org
"""


def get_sorted(df, i):
    """
    Determine the best value from a specified column in a DataFrame.

    This function selects either the maximum or minimum value of a specified column
    based on the input. For specific column names, the maximum value is chosen,
    while for all other columns, the minimum value is selected.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to evaluate.
        i (str): The name of the column to determine the best value from.

    Returns:
        Sorted list of values from the specified column.
    """
    if i in ("f_UMLIP/f_DFT", "T_1/2^term (K)"):
        return sorted(df[i].dropna(), reverse=True)
    return sorted(df[i].dropna())


@callback(
    Output("pbe-graph", "figure"),
    Input("pbe-benchmarks-table", "selected_columns"),
)
def update_pbe_graphs(selected_columns):
    """
    This function is a callback for updating a bar plot figure of performance benchmarks
    based on the selected columns in a Dash DataTable component. It generates and
    configures the bar chart showing benchmark comparisons between architectures for
    various datasets.

    Args:
        selected_columns (list): A list of selected column names from the benchmark
        DataTable. It is expected to be a non-empty list with the first column name
        used for the y-axis of the plot.

    Returns:
        plotly.graph_objects.Figure: A bar chart figure, showing the benchmark
        comparisons using the selected column for the y-axis, grouped by architecture.
    """
    layout = dict(font=dict(size=18))
    col = selected_columns[0]
    fig = px.bar(pbe_df, x="Dataset", y=col, color="Architecture", barmode="group")
    fig.update_layout(**layout)
    return fig


@callback(
    Output("r2scan-graph", "figure"),
    Input("r2scan-benchmarks-table", "selected_columns"),
)
def update_r2scan_graphs(selected_columns):
    """
    Updates the R2SCAN graph figure based on selected benchmark table columns. The function takes
    the selected columns from a table as input and generates a bar graph displaying benchmark metrics
    grouped by architecture, using the provided dataset. The graph figure is updated with a defined
    layout for consistent formatting.

    Args:
        selected_columns (list of str): A list containing the selected column names from the
            benchmarks table.

    Returns:
        plotly.graph_objs._figure.Figure: The updated bar graph figure representing the benchmark
            data using the selected column.
    """
    layout = dict(font=dict(size=18))
    col = selected_columns[0]
    # error_y = f"{col.split(' ')[0]} STDAE" if col.endswith("MAE") else None
    fig = px.bar(
        r2scan_df,
        x="Dataset",
        y=col,
        # error_y=error_y,
        color="Architecture",
        barmode="group",
    )
    fig.update_layout(**layout)
    return fig


def gen_data_table(df, name):
    """
    Generates a Dash DataTable with specific configurations for displaying benchmarking
    data from a Pandas DataFrame. The table filters out certain columns, formats numeric
    data, and applies conditional styling to rows and columns based on specified criteria.

    Parameters:
        df (pd.DataFrame): The Pandas DataFrame containing data to display in the table.
        Columns in the DataFrame will be filtered and styled based on the function's logic.

    Returns:
        dash.dash_table.DataTable: A Dash DataTable object configured with the data, styling,
        and sorting properties derived from the input DataFrame.
    """
    cols = [c for c in df.columns if c if not ("diff" in c or "STDAE" in c)]
    return dash_table.DataTable(
        id=f"{name}-benchmarks-table",
        columns=[
            {
                "name": i,
                "id": i,
                "type": "numeric",
                "selectable": i not in ["Dataset", "Architecture"],
                "format": Format(precision=2, scheme=Scheme.decimal, nully="-"),
            }
            for i in cols
        ],
        data=df.to_dict("records"),
        column_selectable="single",
        selected_columns=["d MAE"],
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "rgb(220, 220, 220)",
            }
        ]
        + [
            {
                "if": {"column_id": i, "row_index": np.where(~df[f"{i.split(' ')[0]} sig_diff_rel"])[0]},
                "font-weight": "bold",
                "color": "white",
                "background-color": "#633D9Caa",
            }
            for i in cols
            if i.endswith("MAE")
        ]
        + [
            {
                "if": {
                    "filter_query": "{{T_1/2^term}} = {}".format(df["T_1/2^term"].max()),
                    "column_id": "T_1/2^term",
                },
                "font-weight": "bold",
                "color": "white",
                "background-color": "#633D9Caa",
            },
            {
                "if": {
                    "filter_query": "{{f/f_DFT}} = {}".format(pbe_df["f/f_DFT"].max()),
                    "column_id": "f/f_DFT",
                },
                "font-weight": "bold",
                "color": "white",
                "background-color": "#633D9Caa",
            },
            {
                "if": {
                    "filter_query": "{{t_step}} = {}".format(df["t_step"].min()),
                    "column_id": "t_step",
                },
                "font-weight": "bold",
                "color": "white",
                "background-color": "#633D9Caa",
            },
        ],
        style_header={"backgroundColor": "#633D9C", "color": "white", "fontWeight": "bold"},
        sort_action="native",
    )


layout = dbc.Container(
    [
        dbc.Col(
            html.Div([dcc.Markdown(INTRO_CONTENT)]),
            width=12,
        ),
        dbc.Col(html.H4("PBE", className="section-title"), width=12),
        dbc.Col(dcc.Graph(id="pbe-graph"), width=12),
        dbc.Col(
            html.Div(
                "Clicking on the radio buttons graphs the selected column.",
            ),
            width=12,
        ),
        dbc.Col(
            gen_data_table(pbe_df, "pbe"),
            width=12,
        ),
        dbc.Col(
            html.Div(dcc.Markdown(TABLE_NOTE)),
            width=12,
        ),
        dbc.Col(
            html.H4("r2SCAN", className="section-title"),
            width=12,
            style={"padding-top": "30px"},
        ),
        dbc.Col(
            html.Div(
                "There are only a limited number of MatPES r2SCAN benchmarks for different models due to"
                " the limited amount of other r2SCAN training data sets and ground-truth r2SCAN DFT data.",
            ),
            width=12,
        ),
        dbc.Col(dcc.Graph(id="r2scan-graph"), width=12),
        dbc.Col(
            html.Div(
                "Clicking on the radio buttons graphs the selected column.",
            ),
            width=12,
        ),
        dbc.Col(
            gen_data_table(r2scan_df, "r2scan"),
            width=12,
        ),
        dbc.Col(
            html.Div(dcc.Markdown(TABLE_NOTE)),
            width=12,
        ),
        dbc.Col(html.H4("Overview of MatCalc-Benchmark Metrics"), width=12, style={"padding-top": "30px"}),
        dbc.Col(
            html.Div([dcc.Markdown(LEGEND)], id="matcalc-benchmark-legend"),
            width=12,
            style={"padding-top": "10px"},
        ),
    ]
)
