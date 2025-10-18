# api.py
from fastapi import FastAPI
import tensorflow as tf
import joblib
import numpy as np

app = FastAPI()

# Load model + scaler
model = tf.keras.models.load_model("diagnosis_ann.h5")
scaler = joblib.load("scaler.pkl")

@app.post("/predict")
def predict(features: list):
    X = np.array(features).reshape(1, -1)
    X = scaler.transform(X)
    prob = model.predict(X)[0][0]
    result = int(prob > 0.5)
    return {"probability": float(prob), "prediction": result}
