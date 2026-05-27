"""
FastAPI application for mining equipment failure prediction.

This API loads the trained Machine Learning model saved in:
    models/failure_prediction_model.pkl

It exposes two endpoints:
    GET  /health   -> checks if the API and model are available
    POST /predict  -> predicts failure risk for one mining equipment record

Run from the project root:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Project paths and model loading
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "failure_prediction_model.pkl"


# These ranges are aligned with the synthetic data generation script.
# They are used to calculate operational_stress_score for new API requests.
SCALING_RANGES = {
    "operation_hours": (100, 25_000),
    "temperature_celsius": (35, 125),
    "vibration_mm_s": (0.5, 18),
    "load_percentage": (15, 100),
    "maintenance_days_since_last": (1, 240),
    "current_amp": (80, 900),
}


def load_model_package(model_path: Path) -> Dict:
    """Load the trained model package from disk.

    The training script saves a dictionary containing:
    - model: trained scikit-learn pipeline
    - model_name: selected model name
    - feature_columns: exact columns used during training
    - target_column: target variable name
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. "
            "Run: python src/models/train_model.py"
        )

    model_package = joblib.load(model_path)

    required_keys = {"model", "model_name", "feature_columns"}
    missing_keys = required_keys.difference(model_package.keys())

    if missing_keys:
        raise ValueError(f"Invalid model package. Missing keys: {missing_keys}")

    return model_package


try:
    MODEL_PACKAGE = load_model_package(MODEL_PATH)
    MODEL = MODEL_PACKAGE["model"]
    MODEL_NAME = MODEL_PACKAGE["model_name"]
    FEATURE_COLUMNS = MODEL_PACKAGE["feature_columns"]
except Exception as error:
    # The API can still start, but /health and /predict will report the issue.
    MODEL_PACKAGE = None
    MODEL = None
    MODEL_NAME = None
    FEATURE_COLUMNS = []
    MODEL_LOADING_ERROR = str(error)
else:
    MODEL_LOADING_ERROR = None

# FastAPI app
tags_metadata = [
    {
        "name": "Mining Equipment Endpoints",  # El nuevo título que reemplazará a 'default'
        "description": "Operaciones principales para el monitoreo y predicción de fallas.",
    },
]
app = FastAPI(
    title="Mining Equipment Failure Prediction API",
    description=(
        "API for predicting the probability of equipment failure in the next "
        "7 days using a trained Machine Learning model."
    ),
    version="1.0.0",
    tags_metadata=tags_metadata,
)

# Pydantic schemas
class EquipmentInput(BaseModel):
    """Input data required to predict equipment failure risk."""

    equipment_type: Literal["truck", "conveyor_belt", "pump", "crusher", "mill"]
    operation_hours: float = Field(..., ge=0, description="Accumulated operation hours")
    temperature_celsius: float = Field(..., description="Equipment temperature in Celsius")
    vibration_mm_s: float = Field(..., ge=0, description="Vibration level in mm/s")
    pressure_bar: float = Field(..., ge=0, description="Operating pressure in bar")
    current_amp: float = Field(..., ge=0, description="Electrical current in amperes")
    load_percentage: float = Field(..., ge=0, le=100, description="Equipment load percentage")
    maintenance_days_since_last: int = Field(
        ..., ge=0, description="Days since the last maintenance activity"
    )
    failure_history_count: int = Field(..., ge=0, description="Previous failure count")
    environment_dust_level: Literal["low", "medium", "high"]
    shift: Literal["day", "night"]


class PredictionOutput(BaseModel):
    """Response returned by the prediction endpoint."""

    failure_probability: float
    risk_level: Literal["low", "medium", "high"]
    recommendation: str


# Feature engineering for inference
def min_max_scale(value: float, min_value: float, max_value: float) -> float:
    """Scale one value between 0 and 1 using predefined ranges."""
    if max_value == min_value:
        return 0.0

    scaled_value = (value - min_value) / (max_value - min_value)

    # Keep the score in a controlled range for values outside the synthetic limits.
    return max(0.0, min(1.0, scaled_value))


def calculate_operational_stress_score(data: Dict) -> float:
    """Calculate the same operational stress score used during data processing."""
    return (
        0.25
        * min_max_scale(data["operation_hours"], *SCALING_RANGES["operation_hours"])
        + 0.25
        * min_max_scale(
            data["temperature_celsius"], *SCALING_RANGES["temperature_celsius"]
        )
        + 0.20
        * min_max_scale(data["vibration_mm_s"], *SCALING_RANGES["vibration_mm_s"])
        + 0.15
        * min_max_scale(data["load_percentage"], *SCALING_RANGES["load_percentage"])
        + 0.10
        * min_max_scale(
            data["maintenance_days_since_last"],
            *SCALING_RANGES["maintenance_days_since_last"],
        )
        + 0.05 * min_max_scale(data["current_amp"], *SCALING_RANGES["current_amp"])
    )


def prepare_features(input_data: EquipmentInput) -> pd.DataFrame:
    """Convert API input into the exact feature format expected by the model."""
    data = input_data.model_dump()

    # Start with all training columns in zero.
    # This prevents missing-column errors during prediction.
    features = {column: 0 for column in FEATURE_COLUMNS}

    # Direct numeric variables.
    numeric_columns = [
        "operation_hours",
        "temperature_celsius",
        "vibration_mm_s",
        "pressure_bar",
        "current_amp",
        "load_percentage",
        "maintenance_days_since_last",
        "failure_history_count",
    ]

    for column in numeric_columns:
        if column in features:
            features[column] = data[column]

    # Risk flags created with the same logic used in Pandas processing.
    if "high_temperature_flag" in features:
        features["high_temperature_flag"] = int(data["temperature_celsius"] >= 85)

    if "high_vibration_flag" in features:
        features["high_vibration_flag"] = int(data["vibration_mm_s"] >= 7.5)

    if "maintenance_risk_flag" in features:
        features["maintenance_risk_flag"] = int(
            data["maintenance_days_since_last"] >= 90
            or data["failure_history_count"] >= 3
        )

    if "operational_stress_score" in features:
        features["operational_stress_score"] = calculate_operational_stress_score(data)

    # One-hot encoded categorical variables.
    equipment_column = f"equipment_type_{data['equipment_type']}"
    dust_column = f"environment_dust_level_{data['environment_dust_level']}"
    shift_column = f"shift_{data['shift']}"

    for column in [equipment_column, dust_column, shift_column]:
        if column in features:
            features[column] = 1

    # Return a DataFrame with the same column order used during training.
    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


# Business rules for risk interpretation
def get_risk_level(probability: float) -> str:
    """Convert failure probability into an easy business risk level."""
    if probability >= 0.65:
        return "high"
    if probability >= 0.30:
        return "medium"
    return "low"


def get_recommendation(risk_level: str) -> str:
    """Return a maintenance recommendation based on the risk level."""
    recommendations = {
        "high": (
            "Programar inspección prioritaria del equipo. Revisar vibración, "
            "temperatura, historial de fallas y condición de mantenimiento antes "
            "de continuar con operación prolongada."
        ),
        "medium": (
            "Monitorear el equipo en la siguiente ronda de mantenimiento. Validar "
            "tendencia de vibración y temperatura para evitar una falla no programada."
        ),
        "low": (
            "Mantener operación normal y continuar con el plan de monitoreo preventivo."
        ),
    }

    return recommendations[risk_level]


# API endpoints
@app.get("/health", tags=["Mining Equipment Endpoints"])
def health_check() -> Dict:
    """Check if the API is running and the model was loaded correctly."""
    if MODEL_LOADING_ERROR:
        return {
            "status": "error",
            "model_loaded": False,
            "message": MODEL_LOADING_ERROR,
        }

    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": MODEL_NAME,
        "feature_count": len(FEATURE_COLUMNS),
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Mining Equipment Endpoints"])
def predict_failure_risk(input_data: EquipmentInput) -> PredictionOutput:
    """Predict failure probability and return a business recommendation."""
    if MODEL is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model is not available. Error: {MODEL_LOADING_ERROR}",
        )

    features = prepare_features(input_data)

    # Most scikit-learn classifiers provide predict_proba.
    # Class 1 represents failure_next_7_days = 1.
    if hasattr(MODEL, "predict_proba"):
        failure_probability = float(MODEL.predict_proba(features)[0][1])
    else:
        prediction = int(MODEL.predict(features)[0])
        failure_probability = float(prediction)

    risk_level = get_risk_level(failure_probability)
    recommendation = get_recommendation(risk_level)

    return PredictionOutput(
        failure_probability=round(failure_probability, 4),
        risk_level=risk_level,
        recommendation=recommendation,
    )