"""
FastAPI application for depression risk prediction.
Provides /predict and /retrain endpoints.
"""
import os
from io import BytesIO
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from prediction import predict

app = FastAPI(
    title="Depression Risk Prediction API",
    description="Predicts depression risk for university students using ML.",
    version="1.0.0",
)

# CORS: Explicit configuration (not generic allow *)
# Add your Render API URL to allow_origins when deployed, e.g. https://your-api.onrender.com
_cors_origins = [
    "https://your-flutter-app.com",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
]
if os.environ.get("RENDER_EXTERNAL_URL"):
    _cors_origins.append(os.environ["RENDER_EXTERNAL_URL"].rstrip("/"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)


class PredictionInput(BaseModel):
    """Input schema for prediction with enforced types and ranges."""

    age: float = Field(..., ge=18, le=59, description="Age in years (18-59)")
    gender: Literal["Male", "Female"] = Field(..., description="Gender")
    academic_pressure: float = Field(..., ge=0, le=5, description="Academic pressure (0-5)")
    work_pressure: float = Field(..., ge=0, le=5, description="Work pressure (0-5)")
    cgpa: float = Field(..., ge=0, le=10, description="CGPA (0-10)")
    study_satisfaction: float = Field(..., ge=0, le=5, description="Study satisfaction (0-5)")
    sleep_duration: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Sleep duration (e.g. '5-6 hours', 'Less than 5 hours')",
    )
    dietary_habits: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Dietary habits (e.g. 'Healthy', 'Moderate')",
    )
    degree: str = Field(..., min_length=1, max_length=50, description="Degree (e.g. 'BSc', 'BE')")
    suicidal_thoughts: Literal["Yes", "No"] = Field(
        ...,
        description="Have you ever had suicidal thoughts?",
    )
    work_study_hours: float = Field(
        ...,
        ge=0,
        le=12,
        description="Work/study hours per day (0-12)",
    )
    financial_stress: float = Field(
        ...,
        ge=1,
        le=5,
        description="Financial stress level (1-5)",
    )
    family_history_mental_illness: Literal["Yes", "No"] = Field(
        ...,
        description="Family history of mental illness",
    )


def _input_to_features(data: PredictionInput) -> dict:
    """Map API input to internal feature names."""
    return {
        "Age": data.age,
        "Gender": data.gender,
        "Academic Pressure": data.academic_pressure,
        "Work Pressure": data.work_pressure,
        "CGPA": data.cgpa,
        "Study Satisfaction": data.study_satisfaction,
        "Sleep Duration": data.sleep_duration,
        "Dietary Habits": data.dietary_habits,
        "Degree": data.degree,
        "Have you ever had suicidal thoughts ?": data.suicidal_thoughts,
        "Work/Study Hours": data.work_study_hours,
        "Financial Stress": data.financial_stress,
        "Family History of Mental Illness": data.family_history_mental_illness,
    }


@app.get("/")
def root():
    """Health check."""
    return {"status": "ok", "message": "Depression Risk Prediction API"}


@app.post("/predict")
def predict_endpoint(data: PredictionInput):
    """
    Predict depression risk score given student features.
    Returns a float (0-1 range; higher = higher risk).
    """
    try:
        features = _input_to_features(data)
        score = predict(features)
        return {"prediction": score, "status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/retrain")
async def retrain_endpoint(file: UploadFile = File(None)):
    """
    Trigger model retraining when new data is uploaded.
    Accepts a CSV file with columns matching the training dataset.
    """
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    if file is None or file.filename == "":
        raise HTTPException(
            status_code=400,
            detail="No file uploaded. Send a CSV file with the same schema as training data.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        # Validate required columns
        required = [
            "Age", "Gender", "Academic Pressure", "Work Pressure", "CGPA",
            "Study Satisfaction", "Sleep Duration", "Dietary Habits", "Degree",
            "Have you ever had suicidal thoughts ?", "Work/Study Hours",
            "Financial Stress", "Family History of Mental Illness", "Depression",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}. CSV must match training schema.",
            )

        feature_cols = [c for c in required if c != "Depression"]
        X = df[feature_cols].copy()
        y = df["Depression"]

        # Fill NaN in Financial Stress
        X["Financial Stress"] = X["Financial Stress"].fillna(X["Financial Stress"].median())

        # Encode categorical
        cat_cols = X.select_dtypes(include="object").columns.tolist()
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            encoders[col] = le
            X[col] = le.transform(X[col].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models and pick best
        models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(max_depth=8, random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        }
        best_name, best_mse = None, float("inf")
        best_model = None

        for name, m in models.items():
            m.fit(X_train_scaled, y_train)
            preds = m.predict(X_test_scaled)
            mse = mean_squared_error(y_test, preds)
            if mse < best_mse:
                best_mse = mse
                best_name = name
                best_model = m

        # Save artifacts to API directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        joblib.dump(best_model, os.path.join(script_dir, "best_model.pkl"))
        joblib.dump(scaler, os.path.join(script_dir, "scaler.pkl"))
        joblib.dump(encoders, os.path.join(script_dir, "encoders.pkl"))
        joblib.dump(feature_cols, os.path.join(script_dir, "feature_cols.pkl"))

        return {
            "status": "success",
            "message": "Model retrained successfully",
            "best_model": best_name,
            "test_mse": round(best_mse, 6),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
