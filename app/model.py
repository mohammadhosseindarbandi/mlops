import joblib
import numpy as np

class HeartDiseaseModel:
    def __init__(self):
        self.model = joblib.load("app/model/model.pkl")
        self.scaler = joblib.load("app/model/scaler.pkl")
        self.selector = joblib.load("app/model/feature_selector.pkl")

    def predict(self, data: np.ndarray):
        scaled = self.scaler.transform(data)
        selected = self.selector.transform(scaled)
        prediction = self.model.predict(selected)
        return int(prediction[0])
