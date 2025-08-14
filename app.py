import json
import pickle
from pathlib import Path

from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model next to this file
model_path = Path(__file__).with_name("linreg_pipeline.pkl")
try:
    with model_path.open("rb") as f:
        regmodel = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {model_path}. Place linreg_pipeline.pkl next to app.py.")

COLUMNS = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify(error="Invalid or missing JSON. Set Content-Type: application/json"), 400

    data = payload.get("data", payload)
    records = [data] if isinstance(data, dict) else data

    try:
        df = pd.DataFrame(records)[COLUMNS]
    except KeyError as e:
        return jsonify(error=f"Missing keys: {e}"), 400

    preds = regmodel.predict(df).tolist()
    return jsonify(predictions=preds)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = {k: float(v) for k, v in request.form.items()}
        df = pd.DataFrame([form])[COLUMNS]
    except Exception as e:
        return render_template("home.html", prediction_text=f"Invalid input: {e}")

    output = regmodel.predict(df)[0]
    return render_template("home.html", prediction_text=f"The House price prediction is {output}")

if __name__ == "__main__":
   app.run(debug=True, host="127.0.0.1", port=5000)