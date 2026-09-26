"""Production entry point for the hosted Dash application."""
from __future__ import annotations

import os

from analysis import build_analysis
from dashboard import create_app


def build_app():
    """Train the analysis and return the configured Dash application."""
    data, rules, explanations = build_analysis()
    return create_app(data, rules, explanations)


# The Dash entry point is retained for local/server deployments; GitHub Pages uses
# the static frontend in docs/ and the export script instead.
app = build_app()
server = app.server


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8050")),
        debug=False,
    )
