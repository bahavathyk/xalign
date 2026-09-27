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

For the default German Credit view, serve `docs/` with any static web server, for example `python -m http.server 8000 --directory docs`, and open `http://localhost:8000`.

For accurate custom CSV analysis, run the local Python API instead:

```text
python local_server.py
```

Then open `http://localhost:8000`. The local server trains a real scikit-learn Random Forest and computes TreeSHAP, KernelSHAP, LIME, and decision-tree rule statistics for the selected CSV columns. Custom uploads are intentionally hidden on the GitHub Pages version, which remains a static German Credit demonstration.

Do not run `python -m http.server` when using CSV uploads; that only serves static files and has no `/api/analyze` endpoint. If port 8000 is already occupied, use `$env:PORT=8001` before starting `local_server.py`, then open `http://localhost:8001`.

The GitHub Actions workflow in `.github/workflows/pages.yml` regenerates the data and publishes `docs/` whenever `main` changes. In the repository settings, set **Pages → Build and deployment → Source** to **GitHub Actions**.

The dashboard compares TreeSHAP, KernelSHAP, and LIME explanations for the UCI German Credit model, with rule counts overlaid through the shared controls.

Custom CSV uploads are available only through `local_server.py`, so the uploaded data remains on the local machine and the analysis uses the real Python XAI pipeline. To keep KernelSHAP and LIME responsive, files larger than 2,000 rows are reproducibly sampled to 2,000 rows before training.
