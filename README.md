# German Credit Visual Analytics Dashboard

Run `pip install -r requirements.txt`, then `python main.py` from this directory.

The dashboard trains a Random Forest on the UCI German Credit data and presents TreeSHAP, KernelSHAP, and LIME in parallel. Categorical variables retain their original feature name and reveal their categories; one-hot encoder column names are never shown. Every explanation panel shares one signed colour scale, and compact rule markers are overlaid without obscuring the heatmap.
