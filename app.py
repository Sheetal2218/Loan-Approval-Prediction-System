# Step 1: Imports
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Step 2: App initialize
app = Flask(__name__)

# Step 3: Model load
saved = joblib.load("model.pkl")
model = saved["model"]
scaler = saved["scaler"]
features = saved["features"]

# Step 4: Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_values = [float(data[feat]) for feat in features]
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        result = {
            "approved": bool(prediction),
            "probability": round(float(probability) * 100, 2)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Step 5: Run server
if __name__ == "__main__":
    app.run(debug=True)