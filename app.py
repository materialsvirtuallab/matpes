"""Main app."""

from __future__ import annotations

import argparse

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html
from dash_bootstrap_templates import load_figure_template

load_figure_template("pulse")

app = Dash(
    "MatPES",
    use_pages=True,
    external_stylesheets=[dbc.themes.PULSE],
    title="MatPES",
)

download_bar = dbc.Row(
    [
        dbc.Col(
            dbc.NavbarBrand("Download", id="download-brand"),
        ),
        dbc.Col(
            dbc.Button(
                "PBE",
                href="https://mavrl-web.s3.us-east-1.amazonaws.com/matpes/MatPES-PBE-20240214.json.gz",
                class_name="ms-2 download-button",
                color="success",
                external_link=True,
            ),
        ),
        dbc.Col(
            dbc.Button(
                "r2SCAN",
                href="https://mavrl-web.s3.us-east-1.amazonaws.com/matpes/MatPES-r2SCAN-20240214.json.gz",
                class_name="ms-2 download-button",
                color="info",
                external_link=True,
            ),
            width="auto",
        ),
    ],
    class_name="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(src=dash.get_asset_url("logo.svg"), alt="MatPES", id="header-logo"),
                    ),
                    dbc.Col(html.A(dbc.NavbarBrand("MatPES.ai", class_name="ms-2"), href="/")),
                ],
                align="center",
                class_name="g-0",
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                [
                    dbc.Row(
                        [dbc.Col(dbc.NavLink("Home", href="/", class_name="ms-4 nav-link-item", active="exact"))]
                        + [
                            dbc.Col(
                                dbc.NavLink(
                                    name, href=f"/{name.lower()}", class_name="ms-4 nav-link-item", active="exact"
                                )
                            )
                            for name in ("Explorer", "Dataset", "Benchmarks")
                        ],
                        align="center",
                        class_name="g-0",
                    ),
                    download_bar,
                ],
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ]
    ),
    color="primary",
    dark=True,
)


content = html.Div(children=dash.page_container, id="page-content")

footer_style = {
    "border-top": "1px solid #111",  # Add a border at the top
    "text-align": "center",  # Center-align the text
    "padding": "10px",  # Add some padding for spacing
    "font-size": "0.8rem",
}

footer = html.Footer(["© ", html.A("Materials Virtual Lab", href="http://materialsvirtuallab.org")], style=footer_style)

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        <!-- Google tag (gtag.js) -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-0P0W73YK15"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());

          gtag('config', 'G-0P0W73YK15');
        </script>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

app.layout = html.Div([dcc.Location(id="url"), navbar, content, footer])


server = app.server


@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    """Toggle navbar collapse on small screens."""
    if n:
        return not is_open
    return is_open


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
