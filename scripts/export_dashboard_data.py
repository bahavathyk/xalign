"""Export the model results used by the static GitHub Pages frontend."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import build_analysis
from explainers import aggregate_explanations
from plotting import numeric_edges
from rules import rule_statistics


OUTPUT = ROOT / "docs" / "data" / "dashboard.json"


def serialise_records(explanation_frames):
    records = []
    for method, frame in explanation_frames.items():
        for row in frame.itertuples(index=False):
            records.append({"method": method, "feature": row.Feature, "category": row.Category,
                            "class": int(row.Class), "bin": int(row.Bin), "importance": float(row.Importance)})
    return records


def serialise_rules(counts):
    return [{"feature": feature, "category": category, "class": int(cls), "bin": int(bin_id), "count": int(count)}
            for (feature, category, bin_id, cls), count in counts.items()]


def main():
    data, rules, explanations = build_analysis()
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
        by_bins[str(bin_count)] = {"records": serialise_records(frames), "rules": serialise_rules(counts)}

    payload = {"schemaVersion": 1, "classes": data.classes, "methods": list(explanations),
               "features": features, "bins": by_bins}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
