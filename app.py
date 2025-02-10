"""Main app."""

from __future__ import annotations

import argparse

import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

app = Dash(
    "MatPES",
    use_pages=True,
    external_stylesheets=[dbc.themes.ZEPHYR],
    title="MatPES",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(html.Img(src=dash.get_asset_url("logo.png"), alt="MatPES", id="header-logo")),
                    dbc.Col(html.A(dbc.NavbarBrand("MatPES", className="ms-2"), href="/")),
                ]
                + [
                    dbc.Col(dbc.NavLink(page["name"], className="ms-2 text-light", href=page["relative_path"]))
                    for page in dash.page_registry.values()
                    # ]
                    # + [
                    #     dbc.Col(dbc.Button("PBE", color="primary", className="me-1")),
                    #     dbc.Col(dbc.Button("R2SCAN", color="secondary", className="me-1")),
                ],
                align="center",
                className="g-0",
                style={"textDecoration": "none"},
            )
        ]
    ),
    color="primary",
    dark=True,
)

content = html.Div(children=dash.page_container, id="page-content")

app.layout = html.Div([dcc.Location(id="url"), navbar, content])

server = app.server


def main():
    """Main entry point for MatPES Webapp."""
    parser = argparse.ArgumentParser(
        description="""MatPES.ai is a Dash Interface for MatPES.""",
        epilog="Author: Shyue Ping Ong",
    )

    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        action="store_true",
        help="Whether to run in debug mode.",
    )
    parser.add_argument(
        "-hh",
        "--host",
        dest="host",
        type=str,
        nargs="?",
        default="0.0.0.0",
        help="Host in which to run the server. Defaults to 0.0.0.0.",
    )
    parser.add_argument(
        "-p",
        "--port",
        dest="port",
        type=int,
        nargs="?",
        default=8050,
        help="Port in which to run the server. Defaults to 8050.",
    )

    args = parser.parse_args()

    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
