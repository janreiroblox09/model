import os
from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Load the trained Random Forest model using the environment variable for the path
model_path = os.getenv("MODEL_PATH", "crop_recommendation_model.pkl")
label_encoder_path = os.getenv("LABEL_ENCODER_PATH", "label_encoder.pkl")
scaler_path = os.getenv("SCALER_PATH", "scaler.pkl")

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Load the label encoder
with open(label_encoder_path, "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load the scaler
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

class AveragesData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    rainfall: float
    soilPH: float

@app.post("/predict/")
def predict_crop(data: AveragesData):
    input_data = np.array([[data.nitrogen, data.phosphorus, data.potassium, data.temperature, data.humidity, data.soilPH, data.rainfall]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    crop = label_encoder.inverse_transform([prediction])[0]
    return {"recommended_crop": crop}
