from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load the trained Random Forest model
with open("crop_recommendation_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the label encoder
with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load the scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define input model
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
