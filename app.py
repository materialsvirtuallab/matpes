"""Main app."""

from __future__ import annotations

import argparse

import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

app = Dash(
    "MatPES",
    use_pages=True,
    external_stylesheets=[dbc.themes.CERULEAN],
    title="MatPES",
)

app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Ul(
                        [
                            html.Li(dcc.Link(page["name"], href=page["relative_path"]))
                            for page in dash.page_registry.values()
                        ]
                    ),
                    width={"size": 6, "offset": 2},
                ),
                dbc.Col(
                    html.A(
                        html.Img(
                            src=dash.get_asset_url("logo.png"),
                            alt="MatPES",
                            id="header-logo",
                        ),
                        href="/",
                    ),
                    width={"size": 2},
                ),
            ],
            id="navbar",
        ),
        dash.page_container,
    ]
)

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
