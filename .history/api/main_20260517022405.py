"""FastAPI app to serve overall_rating predictions.

This app loads the scaler and best model at startup (lifespan), exposes
health and predict endpoints, and uses CORS middleware to allow all origins.
"""

from contextlib import asynccontextmanager
import os
import pickle
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Pydantic model for incoming player features
class PlayerFeatures(BaseModel):
    height_cm: float
    weight_kg: float
    age: float
    pace: float
    shooting: float
    passing: float
    dribbling: float
    defending: float
    physic: float
    value_eur: float
    wage_eur: float
    best_position_encoded: int = Field(..., alias='best_position_encoded')
    preferred_foot_encoded: int = Field(..., alias='preferred_foot_encoded')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to load the scaler and model on startup.

    The loaded objects are attached to `app.state` for access in endpoints.
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    model_path = os.path.join(models_dir, 'best_model.pkl')
    app.state.scaler = None
    app.state.model = None
    # Attempt to load scaler
    try:
        with open(scaler_path, 'rb') as f:
            app.state.scaler = pickle.load(f)
    except Exception:
        app.state.scaler = None
    # Attempt to load model
    try:
        with open(model_path, 'rb') as f:
            app.state.model = pickle.load(f)
    except Exception:
        app.state.model = None

    yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware allowing all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint returning a simple status message."""
    return {"message": "Player Performance Prediction API", "status": "running"}


@app.get("/health")
async def health():
    """Health endpoint indicating whether the model and scaler are loaded."""
    model_loaded = getattr(app.state, 'model', None) is not None
    scaler_loaded = getattr(app.state, 'scaler', None) is not None
    return {"status": "healthy", "model_loaded": model_loaded, "scaler_loaded": scaler_loaded}


@app.post("/predict")
async def predict(features: PlayerFeatures):
    """Predict overall_rating for a player.

    Example request JSON:
    {
      "height_cm": 180.0,
      "weight_kg": 75.0,
      "age": 24.0,
      "pace": 80.0,
      "shooting": 70.0,
      "passing": 72.0,
      "dribbling": 78.0,
      "defending": 60.0,
      "physic": 75.0,
      "value_eur": 5000000.0,
      "wage_eur": 20000.0,
      "best_position_encoded": 5,
      "preferred_foot_encoded": 1
    }

    The endpoint scales the input using the loaded scaler and returns the
    predicted overall_rating rounded to 1 decimal place.
    """
    # Check model and scaler
    model = getattr(app.state, 'model', None)
    scaler = getattr(app.state, 'scaler', None)
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail='Model or scaler not loaded')

    # Build feature vector in the expected order
    try:
        x = np.array([
            features.height_cm,
            features.weight_kg,
            features.age,
            features.pace,
            features.shooting,
            features.passing,
            features.dribbling,
            features.defending,
            features.physic,
            features.value_eur,
            features.wage_eur,
            features.best_position_encoded,
            features.preferred_foot_encoded,
        ], dtype=float)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid input: {e}')

    # Ensure scaler expects same number of features
    n_in = getattr(scaler, 'n_features_in_', None)
    if n_in is not None and n_in != x.reshape(1, -1).shape[1]:
        raise HTTPException(
            status_code=500,
            detail=f'Scaler expects {n_in} features but received {x.reshape(1,-1).shape[1]}. '
                   'Ensure the saved scaler matches the input feature vector.'
        )

    # Scale and predict
    try:
        x_scaled = scaler.transform(x.reshape(1, -1))
        pred = model.predict(x_scaled)
        pred_val = float(pred[0])
        return {"predicted_overall_rating": round(pred_val, 1)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')

