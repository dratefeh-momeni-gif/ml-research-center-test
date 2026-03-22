"""
FastAPI service for research center quality prediction

Author: Atefeh Momeni

Loads trained clustering model and predicts quality tier
for a new research center.
"""

import json
import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel


# Features used in model
FEATURES = [
    "internalFacilitiesCount",
    "hospitals_10km",
    "pharmacies_10km",
    "facilityDiversity_10km",
    "facilityDensity_10km",
]


app = FastAPI(title="Research Center Quality API")


# Load saved model files
kmeans = joblib.load("model/kmeans_model.pkl")
scaler = joblib.load("model/scaler.pkl")

with open("model/cluster_mapping.json", "r") as f:
    cluster_mapping = json.load(f)
    cluster_mapping = {int(k): v for k, v in cluster_mapping.items()}


# Input schema
class CenterInput(BaseModel):
    internalFacilitiesCount: int
    hospitals_10km: int
    pharmacies_10km: int
    facilityDiversity_10km: float
    facilityDensity_10km: float


@app.get("/")
def root():
    return {"status": "API running"}


@app.post("/predict")
def predict(data: CenterInput):

    x = np.array([[
        data.internalFacilitiesCount,
        data.hospitals_10km,
        data.pharmacies_10km,
        data.facilityDiversity_10km,
        data.facilityDensity_10km,
    ]])

    x_scaled = scaler.transform(x)

    cluster = int(kmeans.predict(x_scaled)[0])

    tier = cluster_mapping[cluster]

    return {
        "predictedCategory": tier
    }