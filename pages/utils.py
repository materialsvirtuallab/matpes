"""Utility functions, e.g., pt heatmaps, etc."""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pymatgen.core.periodic_table import Element


@functools.lru_cache
def get_pt_df(include_artificial=False) -> pd.DataFrame:
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
            if (el.name not in ["D", "T"]) and (el.Z <= 92 or include_artificial)
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
        return el.row + 2
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


def pt_heatmap(
    values: dict[str, float], label: str = "value", log: bool = False, include_artificial=False
) -> go.Figure:
    """
    Generate a heatmap visualization of the periodic table.

    Args:
        values (dict[str, float]): Mapping of element symbols to values to visualize.
        label (str): Label for the values displayed.
        log (bool): Whether to use logarithmic scaling for the color axis.
        include_artificial (bool): Whether to include artificial elements. Defaults to False.

    Returns:
        plotly.graph_objects.Figure: A scatter plot representing the heatmap.
    """
    df = get_pt_df(include_artificial=include_artificial)
    df[label] = df["symbol"].map(values) if values else df["X"]
    if log:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            df[f"log10_{label}"] = np.log10(df[label])

    # Initialize periodic table grid
    grid = np.full((9, 18), None, dtype=np.float_)
    label_texts = np.full((9, 18), "", dtype=object)
    hover_texts = np.full((9, 18), "", dtype=object)

    # Fill grid with element symbols, hover text, and category colors
    for _, row in df.iterrows():
        group, period = row["group"], row["period"]
        grid[period - 1, group - 1] = row[label] if not log else row[f"log10_{label}"]
        label_texts[period - 1, group - 1] = f"{row['Z']}<br>{row['symbol']}<br>{row[label]}"
        hover_texts[period - 1, group - 1] = f"{row['Z']}<br>{row['name']}<br>{row[label]}"

    # Create the plot
    fig = go.Figure()

    for el in Element:
        if el.symbol not in values and (el.Z <= 92 or include_artificial):
            fig.add_trace(
                go.Heatmap(
                    z=[-1],
                    x=[get_group(el)],
                    y=[get_period(el)],
                    xgap=1,
                    ygap=1,
                    showscale=False,
                    colorscale="Greys",
                )
            )

    fig.add_trace(
        go.Heatmap(
            z=grid,
            x=list(range(1, 19)),
            y=list(range(1, 9)),
            text=hover_texts,
            hoverinfo="text",
            showscale=True,
            colorscale="matter",
            xgap=1,
            ygap=1,
            coloraxis="coloraxis",
        )
    )

    # Add annotations for element symbols
    for _index, row in df.iterrows():
        group, period = row["group"], row["period"]
        fig.add_annotation(
            x=group,
            y=period,
            text=label_texts[period - 1, group - 1],
            showarrow=False,
            font=dict(
                family="Arial",
                size=14,
                color="black",
                weight="bold",
            ),
            align="center",
        )
    # Hide x-axis
    fig.update_xaxes(showticklabels=False, showgrid=False)

    # Hide y-axis
    fig.update_yaxes(showticklabels=False, showgrid=False)

    # Update layout
    fig.update_layout(
        title=None,
        xaxis=dict(title=None),  # Maintain 1:1 aspect ratio
        yaxis=dict(title=None, scaleanchor="x", scaleratio=1.33, autorange="reversed"),
        width=1200,
        height=900,
    )

    if log:
        max_log = int(df[f"log10_{label}"].max())
        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    title=label,
                    tickmode="array",
                    tickvals=list(range(1, max_log + 1)),
                    ticktext=[f"1e{i}" for i in range(1, max_log + 1)],
                    tickfont=dict(size=14, family="Arial"),
                )
            )
        )

    return fig
