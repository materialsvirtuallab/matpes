"""Main app."""

from __future__ import annotations

import argparse

import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

app = Dash(
    "MatPES Explorer",
    use_pages=True,
    external_stylesheets=[dbc.themes.CERULEAN],
    title="MatPES Explorer",
)

app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            html.Img(
                                src="https://github.com/materialsvirtuallab/matpes/blob"
                                "/2b7f8de716289de8089504a63c6431c456268172/assets/logo.png?raw=true",
                                width="70%",
                                style={
                                    "padding": "12px",
                                },
                            ),
                            className="text-primary text-center",
                        )
                    ],
                    width={"size": 6, "offset": 3},
                ),
            ]
        ),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Ul(
                                [
                                    html.Li(dcc.Link(page["name"], href=page["relative_path"]))
                                    for page in dash.page_registry.values()
                                ]
                            )
                        )
                    ],
                    id="navbar",
                )
            ]
        ),
        # html.H1('Multi-page app with Dash Pages'),
        # html.Div([
        #     html.Div(
        #         dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
        #     ) for page in dash.page_registry.values()
        # ]),
        dash.page_container,
    ]
)

server = app.server


def main():
    """Main entry point for MatPES Webapp."""
    parser = argparse.ArgumentParser(
        description="""MatPES Explorer is a Dash Interface for MatPES.""",
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
