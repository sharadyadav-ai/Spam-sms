from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model & vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI(title="Spam SMS Classifier API")

class SMSInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: SMSInput):
    features = vectorizer.transform([data.text])
    prediction = model.predict(features)[0]
    label = "Spam" if prediction == 1 else "Ham"
    return {"text": data.text, "prediction": label}
