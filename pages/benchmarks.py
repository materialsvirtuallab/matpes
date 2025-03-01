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
## MatCalc-Benchmark

The MatCalc-Benchmark is designed to evaluate the performance of different universal machine learning interatomic
potentials (UMLIPs) on a balanced set of equilibrium, near-equilibrium and molecular dynamics properties of materials.

In addition to the property metrics below, it is important to consider the training data size and the number of
parameters of the UMLIPs. Large datasets are difficult to train with, requiring large amounts of CPU/GPU resources.
For instance, the training time for TensorNet on the MatPES-PBE dataset is about 15 minutes per epoch on a single
Nvidia RTX A6000 GPU, while that for the same model on the OMat24 dataset is around 20 hours per epoch on eight Nvidia
A100 GPUs. UMLIPs with large number of parameters will be expensive to run in MD simulations (see t_step metric below),
limiting the size of the simulation cells or time scales you can study.

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
implement an ASE compatible calculator, which can then be used with the [MatCalc](https://github.com/materialsvirtuallab/matcalc)
package. We will release the equilibrium and near-equilibrium benchmark datasets soon in the
[MatCalc repository](https://github.com/materialsvirtuallab/matcalc) together with benchmarking tools.
"""

LEGEND = r"""
Matcalc-Benchmark metrics can be divided into three categories: equilibrium, near-equilibrium, and molecular dynamics
properties.

| Task                                     | Symbol| Units | Functional   | Test Source   | Number |
|------------------------------------------|-------|-------|--------------|--------------------|--------|
| **Equilibrium**                          |       |       |              |                    |        |
| Structural similarity                    | \|v\| | -     |PBE           | WBM[^1]  | 1, 000  |
|                                          |       | - |  RRSCAN      | GNoME[^2] | 1,000  |
| Formation energy per atom | E_form       | meV/atom | PBE         | WBM                           | 1,000  |
|                                          | meV/atom | RRSCAN      | GNoME                           | 1,000  |
| **Near-equilibrium**                     |             |                                          |        |
| Bulk modulus | K_VRH    | GPa | PBE         | [MP]                   | 3,959  |
| Shear modulus | G_VRH   | GPa| PBE         | [MP]                                       | 3,959  |
| Constant volume heat capacity | C_V |J/mol/K| PBE         | Alexandria[^3]       | 1,170  |
| Off-equilibrium force | \|F_i\| |--| PBE         | WBM high energy states[^4] | 979    |
| **Molecular dynamics**                   |             |                                          |        |
| Median termination temp | T_{1/2}^term | K |  PBE & RRSCAN | [MVL]                              | 172    |
| Ionic conductivity | \sigma        | mS/cm | PBE         | [MVL]                                | 698    |
| Time per atom per time step | t_step|  ms/step/atom | PBE & RRSCAN | [MVL]                                | 1      |

The time per atom per time step (t_step) was computed using MD simulations conducted on a single Intel Xeon Gold core
for a system of 64 Si atoms under ambient conditions (300 K and 1 bar) over 50 ps with a 1 fs time step.

[^1]: Wang, H.-C.; Botti, S.; Marques, M. A. L. Predicting stable crystalline compounds using chemical similarity. npj Computational Materials 2021, 7, 1–9.
[^2]: Merchant, A.; Batzner, S.; Schoenholz, S. S.; Aykol, M.; Cheon, G.; Cubuk, E. D. Scaling deep learning for materials discovery. Nature 2023, 624, 80–85.
[^3] Loew, A.; Sun, D.; Wang, H.-C.; Botti, S.; Marques, M. A. L. Universa Machine Learning Interatomic Potentials are Ready for Phonons. 2024; Arxiv: 2412.16551.
[^4]: Deng, B.; Choi, Y.; Zhong, P.; Riebesell, J.; Anand, S.; Li, Z.; Jun, K.; Persson, K. A.; Ceder, G. Overcoming systematic softening in universal machine learning interatomic potentials by fine-tuning. 2024; Arxiv:2405.07105.

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


def create_graphs(df):
    """
    Creates a series of bar graphs for a given DataFrame, where each graph corresponds
    to a specific column (starting from the third column) of the DataFrame. The charts
    represent the data grouped by "Architecture" and categorized by "Dataset".

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing the data to be visualized.
                               The DataFrame must include columns "Dataset",
                               "Architecture", and additional numerical columns
                               starting from the third column.

    Returns:
        dash.development.base_component.Component: A Dash Bootstrap Component (dbc.Row),
                                                   which contains multiple dbc.Col
                                                   elements, each holding a Dash Core
                                                   Component (dcc.Graph) object. These
                                                   graphs represent the bar charts of
                                                   the DataFrame's numerical columns.
    """
    import plotly.express as px

    cols = []
    for i in df.columns[2:]:
        fig = px.bar(df, x="Dataset", y=i, color="Architecture", barmode="group")
        cols.append(
            dbc.Col(
                dcc.Graph(
                    id=f"{i}_hist",
                    figure=fig,
                ),
                width=6,
            )
        )
    return dbc.Row(cols)


table_styles = []


layout = dbc.Container(
    [
        dbc.Col(
            html.Div([dcc.Markdown(INTRO_CONTENT)]),
            width=12,
        ),
        dbc.Col(html.H4("PBE", className="section-title"), width=12),
        create_graphs(pbe_df),
        dbc.Col(
            dash_table.DataTable(
                id="pbe-benchmarks-table",
                columns=[{"name": i.split("(")[0] if "log" not in i else i, "id": i} for i in pbe_df.columns],
                data=pbe_df.to_dict("records"),
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(220, 220, 220)",
                    }
                ]
                + [
                    {
                        "if": {
                            "filter_query": f"{{{i}}} = {get_sorted(pbe_df, i)[0]}",
                            "column_id": i,
                        },
                        "font-weight": "bold",
                        "color": "white",
                        "background-color": "green",
                    }
                    for i in pbe_df.columns[2:]
                ]
                + [
                    {
                        "if": {
                            "filter_query": f"{{{i}}} = {get_sorted(pbe_df, i)[1]}",
                            "column_id": i,
                        },
                        "font-weight": "bold",
                        "color": "white",
                        "background-color": "#50C878",
                    }
                    for i in pbe_df.columns[2:]
                ],
                style_header={"backgroundColor": "rgb(210, 210, 210)", "color": "black", "fontWeight": "bold"},
                sort_action="native",
            ),
            width=12,
        ),
        dbc.Col(
            html.H4("r2SCAN", className="section-title"),
            width=12,
            style={"padding-top": "30px"},
        ),
        create_graphs(r2scan_df),
        dbc.Col(
            dash_table.DataTable(
                id="r2scan-benchmarks-table",
                columns=[{"name": i.split("(")[0] if "log" not in i else i, "id": i} for i in r2scan_df.columns],
                data=r2scan_df.to_dict("records"),
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(220, 220, 220)",
                    }
                ]
                + [
                    {
                        "if": {
                            "filter_query": f"{{{i}}} = {get_sorted(r2scan_df, i)[0]}",
                            "column_id": i,
                        },
                        "font-weight": "bold",
                        "color": "white",
                        "background-color": "green",
                    }
                    for i in r2scan_df.columns[2:]
                ]
                + [
                    {
                        "if": {
                            "filter_query": f"{{{i}}} = {get_sorted(r2scan_df, i)[1]}",
                            "column_id": i,
                        },
                        "font-weight": "bold",
                        "color": "white",
                        "background-color": "#50C878",
                    }
                    for i in r2scan_df.columns[2:]
                ],
                style_header={"backgroundColor": "rgb(210, 210, 210)", "color": "black", "fontWeight": "bold"},
                sort_action="native",
            ),
            width=6,
        ),
        dbc.Col(html.H4("Overview of Matcalc-Benchmark Metrics"), width=12, style={"padding-top": "30px"}),
        dbc.Col(
            html.Div([dcc.Markdown(LEGEND)]),
            width=12,
            style={"padding-top": "10px"},
        ),
    ]
)
