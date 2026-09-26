"""Dash layout and callbacks; all computation is supplied by main.py."""
from __future__ import annotations

import dash
from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc

from explainers import aggregate_explanations
from plotting import fig_to_src, global_limit, numeric_edges, plot_feature_comparison, plot_numeric_features_comparison, plot_overview
from rules import rule_statistics


def create_app(data, rules, explanation_results):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    categorical = [name for name, info in data.metadata.items() if info.kind == "categorical"]

    def prepared(bin_count):
        edges = numeric_edges(data, bin_count)
        explanations = {name: aggregate_explanations(result, data, edges) for name, result in explanation_results.items()}
        counts, _ = rule_statistics(rules, data.metadata, data.encoded_feature_names, edges)
        return explanations, counts

    initial, counts = prepared(10); limit = global_limit(initial)
    app.layout = dbc.Container([
        html.H2("German Credit Visual Analytics"),
        html.P("Random-forest rules overlaid on a shared-scale TreeSHAP, KernelSHAP, and LIME comparison."),
        dbc.Row([dbc.Col([html.Label("Predicted class"), dcc.Dropdown(id="class", options=[{"label": str(c), "value": c} for c in data.classes], value=data.classes[0], clearable=False)]),
                 dbc.Col([html.Label("Numeric bins"), dcc.Slider(id="bins", min=3, max=30, step=1, value=10, marks={x: str(x) for x in (3, 10, 20, 30)})])]),
        dbc.Tabs([
            dbc.Tab(html.Img(id="overview", style={"maxWidth": "100%"}), label="Overview"),
            dbc.Tab([dcc.Dropdown(id="categorical-feature", options=categorical, value=categorical[0], clearable=False, className="mt-3"), html.Img(id="categorical", style={"maxWidth": "100%"})], label="Categorical Features"),
            dbc.Tab(html.Img(id="numeric", style={"maxWidth": "100%"}), label="Numerical Features"),
        ]),
    ], fluid=True)

    @app.callback(Output("overview", "src"), Input("class", "value"), Input("bins", "value"))
    def update_overview(cls, bin_count):
        explanations, _ = prepared(bin_count)
        return fig_to_src(plot_overview(explanations, data.metadata, cls, global_limit(explanations)))

    @app.callback(Output("categorical", "src"), Input("categorical-feature", "value"), Input("class", "value"), Input("bins", "value"))
    def update_categorical(feature, cls, bin_count):
        explanations, rule_counts = prepared(bin_count)
        return fig_to_src(plot_feature_comparison(feature, data.metadata, explanations, rule_counts, cls, global_limit(explanations)))

    @app.callback(Output("numeric", "src"), Input("class", "value"), Input("bins", "value"))
    def update_numeric(cls, bin_count):
        explanations, rule_counts = prepared(bin_count)
        return fig_to_src(plot_numeric_features_comparison(data.metadata, explanations, rule_counts, cls, global_limit(explanations)))
    return app
