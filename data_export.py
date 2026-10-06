"""Serialize model explanations into the JSON contract used by the frontend."""
from __future__ import annotations

import json
import math

from explainers import aggregate_explanations
from plotting import numeric_edges
from rules import rule_statistics


def serialise_records(explanation_frames):
    records = []
    for method, frame in explanation_frames.items():
        for row in frame.itertuples(index=False):
            category = row.Category
            if category != category:
                category = None
            importance = float(row.Importance)
            records.append({"method": method, "feature": row.Feature, "category": category,
                            "class": int(row.Class), "bin": int(row.Bin),
                            "importance": importance if math.isfinite(importance) else 0.0})
    return records


def serialise_rules(counts):
    return [{"feature": feature, "category": category, "class": int(cls), "bin": int(bin_id), "count": int(count)}
            for (feature, category, bin_id, cls), count in counts.items()]


def build_dashboard_payload(data, rules, explanations):
    features = []
    for name, info in data.metadata.items():
        features.append({"name": name, "kind": info.kind,
                         "categories": [info.label_for(column) for column in info.encoded_columns]
                         if info.kind == "categorical" else []})
    by_bins = {}
    for bin_count in range(3, 31):
        edges = numeric_edges(data, bin_count)
        frames = {name: aggregate_explanations(result, data, edges) for name, result in explanations.items()}
        counts, _ = rule_statistics(rules, data.metadata, data.encoded_feature_names, edges)
        by_bins[str(bin_count)] = {
            "records": serialise_records(frames),
            "rules": serialise_rules(counts),
            "ranges": {feature: [float(values[0]), float(values[-1])] for feature, values in edges.items()},
        }
    return {"schemaVersion": 1, "classes": data.classes, "classLabels": data.class_labels or [str(c) for c in data.classes],
            "methods": list(explanations), "features": features, "bins": by_bins}


def payload_json(data, rules, explanations):
    return json.dumps(build_dashboard_payload(data, rules, explanations), separators=(",", ":"), allow_nan=False)
