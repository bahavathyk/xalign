"""Production entry point for the hosted Dash application."""
from __future__ import annotations

import os

from dashboard import create_app
from explainers import compute_kernel_shap, compute_lime, compute_tree_shap
from preprocessing import train_model
from rules import extract_rules


def build_app():
    """Train the analysis and return the configured Dash application."""
    data = train_model()
    rules = extract_rules(data.pipeline)
    explanations = {"TreeSHAP": compute_tree_shap(data)}
    for name, factory in (("KernelSHAP", compute_kernel_shap), ("LIME", compute_lime)):
        try:
            explanations[name] = factory(data)
        except Exception as error:
            raise RuntimeError(f"{name} could not be computed; all three methods are required.") from error
    return create_app(data, rules, explanations)


# Gunicorn/Render imports this WSGI object as ``app:server``.
app = build_app()
server = app.server


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8050")),
        debug=False,
    )
