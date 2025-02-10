"""Main app."""

from __future__ import annotations

import argparse

import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

app = Dash(
    "MatPES",
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY],
    title="MatPES",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Nav(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=dash.get_asset_url("logo.png"), alt="MatPES", id="header-logo")),
                        dbc.Col(html.A(dbc.NavbarBrand("MatPES", className="ms-2"), href="/")),
                        dbc.Col(dbc.NavLink("Home", className="ms-2", href="/", active="exact")),
                        dbc.Col(dbc.NavLink("Explorer", className="me-auto", href="/explorer", active="exact")),
                        dbc.Col(
                            dbc.Button(
                                "PBE",
                                href="matpes_pbe.json.gz",
                                external_link=True,
                                color="success",
                                className="ms-2",
                                disabled=True,
                            )
                        ),
                        dbc.Col(
                            dbc.Button(
                                "R2SCAN",
                                href="matpes_r2scan.json.gz",
                                external_link=True,
                                color="warning",
                                className="ms-2",
                                disabled=True,
                            )
                        ),
                    ],
                    align="center",
                    className="g-0",
                    style={"textDecoration": "none"},
                ),
                navbar=True,
            ),
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
