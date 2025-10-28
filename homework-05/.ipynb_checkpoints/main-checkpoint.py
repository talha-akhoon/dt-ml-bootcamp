from fastapi import FastAPI
import pickle

app = FastAPI()

with open('pipeline_v2.bin', 'rb') as f:
    pipeline = pickle.load(f)

@app.post("/predict")
def predict(client: dict):
    """
    Predict conversion probability for a client
    """
    # Make prediction
    prediction_proba = pipeline.predict_proba([client])[0, 1]
    
    return {
        "conversion_probability": float(prediction_proba),
        "conversion_probability_rounded": round(float(prediction_proba), 3)
    }

