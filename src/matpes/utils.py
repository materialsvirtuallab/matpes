import plotly.express as px
import pandas as pd
from pymatgen.core.periodic_table import Element, _pt_data
import functools
import numpy as np
import warnings
# Define periodic table data

@functools.lru_cache
def get_pt_df():
    """
    Returns a DataFrame with PT data.
    """
    with warnings.catch_warnings():
        df = pd.DataFrame([{"symbol": el.symbol, "name": el.long_name, "Z": el.Z, "X": el.X, "group": get_group(el),
                            "period": get_period(el), "category": get_category(el)} for el in Element])

        # Create hover text for each element
        df["label"] = df.apply(
            lambda row: f"{row['Z']}<br>{row['symbol']}",
            axis=1
        )

    return df


def get_period(el):
    """
    Special handling of period for pt plotting of rare earths.
    """
    if el.is_actinoid or el.is_lanthanoid:
        return el.row + 3
    return el.row


def get_group(el):
    """
    Special handling of group for pt plotting of rare earths.
    """
    if el.is_actinoid:
        return el.group + el.Z - 89
    if el.is_lanthanoid:
        return el.group + el.Z - 57
    return el.group


def get_category(el):
    if el.Z > 92:
        return "transuranic"
    for cat in ["alkali", "alkaline", "actinoid", "lanthanoid", "halogen", "noble_gas", "metal", "chalcogen"]:
        if getattr(el, f"is_{cat}"):
            return cat
    return ""


def get_pt_heatmap(values, label="value", log=False):
    """
    Args:
        values (dict[str, float]): Values to plot.
        label (str): Label for values.
    """
    df = get_pt_df()
    df[label] = df.apply(
        lambda row: values.get(row['symbol'], 0),
        axis=1
    )
    if log:
        df[f"log10_{label}"] = np.log10(df[label])



    # Create the plot
    fig = px.scatter(
        df,
        x="group",
        y="period",
        color=label if not log else f"log10_{label}",
        text="label",
        # hover_name=label,
        hover_data={"Z": False, "name": False, "label": False, label: True, "X": False, "group": False,
                    "period": False, f"log10_{label}": False},
        color_continuous_scale=px.colors.sequential.Viridis,
    )

    fig.update_traces(marker=dict(
        symbol='square',
        size=40,
        line=dict(color="black", width=1),
    ))

    # Update layout
    fig.update_layout(
        xaxis=dict(title="Group", range=[0.5, 18.5], dtick=1),
        yaxis=dict(title="Period", range=[0.5, 10.5], dtick=1, autorange="reversed"),
        showlegend=False,
        plot_bgcolor="white",
        width=1100,
        height=650,
        font=dict(
            family='Arial',
            size=14,
            color='black',
            weight='bold'  # Make the text bold
        )
    )
    if log:
        max_log = int(df[f"log10_{label}"].max())
        fig.update_layout(coloraxis_colorbar=dict(
            title=label,
            tickvals=list(range(1, max_log+1)),
            ticktext=[f"10^{i}" for i in range(1, max_log+1)]
        ))
    return fig
