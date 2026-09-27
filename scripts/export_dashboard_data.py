"""Export the model results used by the static GitHub Pages frontend."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import build_analysis
from data_export import payload_json


OUTPUT = ROOT / "docs" / "data" / "dashboard.json"


def main():
    data, rules, explanations = build_analysis()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(payload_json(data, rules, explanations), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
