import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load(r"C:\Users\Dell\OneDrive\ドキュメント\python\sentiment_model.pkl")

class InputText(BaseModel):
    text: str

label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API running"}

@app.post("/predict")
def predict(data: InputText):
    text = data.text

    pred = model.predict([text])[0]

    sentiment = label_map.get(pred, "Unknown")

    return {
        "text": text,
        "prediction": sentiment
    }