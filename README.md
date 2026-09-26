# German Credit Visual Analytics Dashboard

## Interactive dashboard

Open the hosted Dash visualisation: **[XAlign interactive dashboard](https://xalign-dashboard.onrender.com)**

The dashboard is hosted as a Python web service, so the link above opens the live interactive app in a browser. The first request on Render's free plan may take a little longer while the service wakes up.

To deploy your own instance on Render:

1. Create a new Render Web Service from this GitHub repository.
2. Render will read [`render.yaml`](render.yaml), install the dependencies, and start the app with Gunicorn.
3. Replace the dashboard URL above with the URL Render gives your service.

Run `pip install -r requirements.txt`, then `python main.py` from this directory.

The dashboard trains a Random Forest on the UCI German Credit data and presents TreeSHAP, KernelSHAP, and LIME in parallel. Categorical variables retain their original feature name and reveal their categories; one-hot encoder column names are never shown. Every explanation panel shares one signed colour scale, and compact rule markers are overlaid without obscuring the heatmap.
