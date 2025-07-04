from fastapi import APIRouter
from app.model import HeartDiseaseModel
from app.schemas import Heart_features
import numpy as np

router = APIRouter()

model = HeartDiseaseModel()

@router.post("/predict")
def predict(data: Heart_features):
    input_data = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])
    prediction = model.predict(input_data)
    return {"prediction": prediction}
    