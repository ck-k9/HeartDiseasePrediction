import os
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../best_model.pkl")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [float(request.form[key]) for key in request.form.keys()]
        data = np.array(data).reshape(1, -1)
        prediction = model.predict(data)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        return render_template("index.html", prediction=result)
    except Exception as e:
        return render_template("index.html", prediction="Error: " + str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
