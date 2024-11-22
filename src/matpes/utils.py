"""Utility functions, e.g., pt heatmaps, etc."""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
from pymatgen.core.periodic_table import Element


@functools.lru_cache
def get_pt_df() -> pd.DataFrame:
    """
    Creates a DataFrame containing periodic table data.

    Returns:
        pd.DataFrame: DataFrame with element data including symbol, name, atomic number (Z),
                      electronegativity (X), group, period, category, and hover label.
    """
    with warnings.catch_warnings():
        # Suppress pymatgen warnings
        warnings.simplefilter("ignore")
        elements = [
            {
                "symbol": el.symbol,
                "name": el.long_name,
                "Z": el.Z,
                "X": el.X,
                "group": get_group(el),
                "period": get_period(el),
                "category": get_category(el),
            }
            for el in Element
            if el.name not in ["D", "T"]
        ]
    df = pd.DataFrame(elements)
    df["label"] = df.apply(lambda row: f"{row['Z']}<br>{row['symbol']}", axis=1)
    return df


def get_period(el: Element) -> int:
    """
    Determine the period for an element, with adjustments for rare earths.

    Args:
        el (Element): An element instance.

    Returns:
        int: The adjusted period number.
    """
    if el.is_actinoid or el.is_lanthanoid:
        return el.row + 3
    return el.row


def get_group(el: Element) -> int:
    """
    Determine the group for an element, with adjustments for rare earths.

    Args:
        el (Element): An element instance.

    Returns:
        int: The adjusted group number.
    """
    if el.is_actinoid:
        return el.group + el.Z - 89
    if el.is_lanthanoid:
        return el.group + el.Z - 57
    return el.group


def get_category(el: Element) -> str:
    """
    Categorize the element based on its type.

    Args:
        el (Element): An element instance.

    Returns:
        str: The category of the element (e.g., alkali, noble_gas, etc.).
    """
    if el.Z > 92:
        return "transuranic"
    for category in [
        "alkali",
        "alkaline",
        "actinoid",
        "lanthanoid",
        "halogen",
        "noble_gas",
        "metal",
        "chalcogen",
    ]:
        if getattr(el, f"is_{category}"):
            return category
    return "other"


def pt_heatmap(values: dict[str, float], label: str = "value", log: bool = False) -> px.scatter:
    """
    Generate a heatmap visualization of the periodic table.

    Args:
        values (dict[str, float]): Mapping of element symbols to values to visualize.
        label (str): Label for the values displayed.
        log (bool): Whether to use logarithmic scaling for the color axis.

    Returns:
        plotly.graph_objects.Figure: A scatter plot representing the heatmap.
    """
    df = get_pt_df()
    df[label] = df["symbol"].map(values)
    if log:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            df[f"log10_{label}"] = np.log10(df[label])

    fig = px.scatter(
        df,
        x="group",
        y="period",
        color=label if not log else f"log10_{label}",
        text="label",
        hover_data={
            "Z": False,
            "name": False,
            "label": False,
            label: True,
            "X": False,
            "group": False,
            "period": False,
            f"log10_{label}": False,
        },
        color_continuous_scale=px.colors.sequential.Viridis,
    )

    fig.update_traces(
        marker=dict(
            symbol="square",
            size=40,
            line=dict(color="black", width=1),
        )
    )

    fig.update_layout(
        xaxis=dict(title=None, range=[0.5, 18.5], dtick=1),
        yaxis=dict(title=None, range=[0.5, 10.5], dtick=1, autorange="reversed"),
        showlegend=False,
        plot_bgcolor="white",
        width=1080,
        height=640,
        font=dict(
            family="Arial",
            size=14,
            color="black",
            weight="bold",
        ),
    )

    # Hide x-axis
    fig.update_xaxes(showticklabels=False, showgrid=False)

    # Hide y-axis
    fig.update_yaxes(showticklabels=False, showgrid=False)

    if log:
        max_log = int(df[f"log10_{label}"].max())
        fig.update_layout(
            coloraxis_colorbar=dict(
                title=label,
                tickvals=list(range(1, max_log + 1)),
                ticktext=[f"10^{i}" for i in range(1, max_log + 1)],
            )
        )
    return fig
