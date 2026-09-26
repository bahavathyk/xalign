# German Credit Visual Analytics Dashboard

## Interactive dashboard

After GitHub Pages is enabled, open the browser-based dashboard at:

**https://bahavathyk.github.io/xalign/**

The static frontend runs entirely in the browser. Python is used only during the GitHub Actions build to train the model and export the compact analysis data consumed by the JavaScript application.

## Local development

Install the Python dependencies and generate the static dataset:

```text
pip install -r requirements.txt
python scripts/export_dashboard_data.py
```

Then serve `docs/` with any static web server, for example `python -m http.server 8000 --directory docs`, and open `http://localhost:8000`.

The GitHub Actions workflow in `.github/workflows/pages.yml` regenerates the data and publishes `docs/` whenever `main` changes. In the repository settings, set **Pages → Build and deployment → Source** to **GitHub Actions**.

The dashboard compares TreeSHAP, KernelSHAP, and LIME explanations for the UCI German Credit model, with rule counts overlaid through the shared controls.
