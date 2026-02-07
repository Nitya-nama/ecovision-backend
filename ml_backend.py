

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from io import BytesIO

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

import requests

app = Flask(__name__)
@app.route("/")
def health():
    return {"message": "EcoVision backend is live"}
CORS(app)
# --- Load dataset safely in cloud ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "gdpdataset_cleaned.csv"))

# ---------------- Countries ----------------
@app.route("/countries")
def countries():
    return jsonify({"countries": sorted(df["country"].dropna().unique().tolist())})

# ---------------- Prediction ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    country = data["country"]
    params = data["parameters"]
    algo = data["algorithm"]

    cdf = df[df["country"].str.lower().str.strip() == country.lower().strip()]
    if cdf.empty:
        return jsonify({"error": "Country not found"}), 400

    X = cdf["year"].values.reshape(-1,1)

    predictions = {}
    metrics = {}

    for param in params:
        if param not in cdf.columns:
            continue

        y = cdf[param].dropna().values
        X_valid = X[:len(y)]

        if algo == "decision_tree":
            model = DecisionTreeRegressor()
        elif algo == "random_forest":
            model = RandomForestRegressor(n_estimators=100)
        elif algo == "svm":
            model = SVR()
        else:
            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(X_valid)
            model = LinearRegression()
            model.fit(X_poly, y)
            y_pred = model.predict(X_poly)
            predictions[param] = y_pred.tolist()
            metrics[param] = {
                "r2": r2_score(y, y_pred),
                "mae": mean_absolute_error(y, y_pred),
                "rmse": np.sqrt(mean_squared_error(y, y_pred))
            }
            continue

        model.fit(X_valid, y)
        y_pred = model.predict(X_valid)

        predictions[param] = y_pred.tolist()
        metrics[param] = {
            "r2": r2_score(y, y_pred),
            "mae": mean_absolute_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred))
        }

    return jsonify({
        "years": cdf["year"].tolist(),
        "predictions": predictions,
        "metrics": metrics
    })

# ---------------- CSV Export ----------------
@app.route("/export/csv")
def export_csv():
    country = request.args.get("country")
    params = request.args.getlist("parameters")

    cdf = df[df["country"].str.lower().str.strip() == country.lower().strip()]
    data = cdf[["year"] + params].dropna()

    buffer = BytesIO()
    data.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(buffer, mimetype="text/csv", as_attachment=True,
                     download_name=f"{country}.csv")
@app.route("/summary", methods=["POST"])
def summary():
    data = request.json

    country = data["country"]
    parameters = data["parameters"]
    years = data["years"]
    predictions = data["predictions"]
    chart = data["chart"]

    preview = ", ".join(map(str, years[:6]))

    lines = []
    for p in parameters:
        vals = predictions.get(p, [])[:6]
        vals = ", ".join([str(round(v,2)) for v in vals])
        lines.append(f"{p}: {vals}")

    prompt = f"""
Provide 3 short insights and 3 suggestions.
Country: {country}
Chart: {chart}
Years: {preview}
Data:
{chr(10).join(lines)}
"""

    API_KEY = os.environ.get("GEMINI_KEY")

    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}",
        json={"contents":[{"role":"user","parts":[{"text":prompt}]}]}
    )

    return jsonify(r.json())

# ---------------- PDF Export ----------------
@app.route("/export/pdf")
def export_pdf():
    country = request.args.get("country")
    params = request.args.getlist("parameters")

    cdf = df[df["country"].str.lower().str.strip() == country.lower().strip()]

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"Economic Report: {country}", styles['h1']), Spacer(1, 0.3*inch)]

    for param in params:
        if param not in cdf.columns:
            continue

        filtered = cdf[["year", param]].dropna()
        plt.figure()
        plt.plot(filtered["year"], filtered[param])
        plt.title(param)

        img = BytesIO()
        plt.savefig(img, format="PNG")
        img.seek(0)
        plt.close()

        story.append(Image(img, width=6*inch, height=3*inch))
        story.append(Spacer(1, 0.2*inch))

    doc.build(story)
    buffer.seek(0)

    return send_file(buffer, mimetype="application/pdf", as_attachment=True,
                     download_name=f"{country}.pdf")

if __name__ == "__main__":
    app.run()
