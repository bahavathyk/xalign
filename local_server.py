"""Local-only server for accurate custom CSV analysis.

Run with: python local_server.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from data_export import build_dashboard_payload
from explainers import compute_kernel_shap, compute_lime, compute_tree_shap
from preprocessing import train_model_from_dataframe
from rules import extract_rules


ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / "docs"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
MAX_ANALYSIS_ROWS = 2000


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/analyze")
def analyze_upload():
    upload = request.files.get("file")
    target = request.form.get("target", "")
    features = [value for value in request.form.getlist("features") if value]
    if upload is None or not target or not features:
        return jsonify(error="A CSV file, target column, and at least one feature are required."), 400
    try:
        frame = pd.read_csv(upload)
        if target not in frame.columns or any(feature not in frame.columns for feature in features):
            return jsonify(error="The selected target or feature column is not present in the CSV."), 400
        if len(frame) > MAX_ANALYSIS_ROWS:
            frame = frame.sample(MAX_ANALYSIS_ROWS, random_state=42)
        data = train_model_from_dataframe(frame, target, features)
        rules = extract_rules(data.pipeline)
        explanations = {"TreeSHAP": compute_tree_shap(data)}
        explanations["KernelSHAP"] = compute_kernel_shap(data, nsamples=100)
        explanations["LIME"] = compute_lime(data, num_samples=1000)
        return jsonify(build_dashboard_payload(data, rules, explanations))
    except Exception as error:
        app.logger.exception("Custom analysis failed")
        return jsonify(error=f"Analysis failed: {error}"), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8000")), debug=False)
