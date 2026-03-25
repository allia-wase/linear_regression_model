import os
import io
import joblib
import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Student Depression Prediction API",
    description=(
        "Predicts depression risk among university students using lifestyle "
        "and academic indicators. Built with Linear Regression."
    ),
    version="1.0.0",
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
# Explicitly configured — NOT using wildcard (*) for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8080",
        "http://localhost:3000",
        "http://10.0.2.2:8000",
        "https://your-app.onrender.com",  # replace with your Render URL
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# ── Load model and scaler ─────────────────────────────────────────────────────
# All paths are relative to wherever prediction.py lives on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"best_model.pkl not found at {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise RuntimeError(f"scaler.pkl not found at {SCALER_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── Encoding maps (match exactly what LabelEncoder produced in the notebook) ──
# These mirror the LabelEncoder fit on the training data.
# Integers are passed directly from the Flutter app — no re-encoding needed.
#
# Sleep Duration:   0=Less than 5hrs, 1=5-6hrs, 2=7-8hrs, 3=More than 8hrs, 4=Others
# Dietary Habits:   0=Healthy, 1=Moderate, 2=Unhealthy
# Gender:           0=Female, 1=Male
# Suicidal Thoughts:0=No, 1=Yes
# Family History:   0=No, 1=Yes
# Degree:           0-20 (LabelEncoded in notebook)
#
# The scaler was fit on numeric data AFTER encoding, so the API
# receives pre-encoded integers directly — no encoders.pkl required.

# ── Input schema ──────────────────────────────────────────────────────────────
class StudentInput(BaseModel):
    Age: float = Field(
        ..., ge=15.0, le=60.0,
        description="Age of the student (15–60)"
    )
    Gender: int = Field(
        ..., ge=0, le=1,
        description="0 = Female, 1 = Male"
    )
    Academic_Pressure: float = Field(
        ..., ge=0.0, le=5.0,
        description="Academic pressure score (0–5)"
    )
    Work_Pressure: float = Field(
        ..., ge=0.0, le=5.0,
        description="Work pressure score (0–5)"
    )
    CGPA: float = Field(
        ..., ge=0.0, le=10.0,
        description="Cumulative GPA (0.0–10.0)"
    )
    Study_Satisfaction: float = Field(
        ..., ge=0.0, le=5.0,
        description="Study satisfaction score (0–5)"
    )
    Sleep_Duration: int = Field(
        ..., ge=0, le=4,
        description="0=<5hrs  1=5-6hrs  2=7-8hrs  3=>8hrs  4=Others"
    )
    Dietary_Habits: int = Field(
        ..., ge=0, le=2,
        description="0=Healthy  1=Moderate  2=Unhealthy"
    )
    Degree: int = Field(
        ..., ge=0, le=20,
        description="Degree type (LabelEncoded integer from training, 0–20)"
    )
    Suicidal_Thoughts: int = Field(
        ..., ge=0, le=1,
        description="0 = No, 1 = Yes"
    )
    Work_Study_Hours: float = Field(
        ..., ge=0.0, le=24.0,
        description="Daily work/study hours (0–24)"
    )
    Financial_Stress: float = Field(
        ..., ge=1.0, le=5.0,
        description="Financial stress score (1–5)"
    )
    Family_History: int = Field(
        ..., ge=0, le=1,
        description="Family history of mental illness: 0 = No, 1 = Yes"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 24.0,
                "Gender": 0,
                "Academic_Pressure": 3.0,
                "Work_Pressure": 1.0,
                "CGPA": 7.5,
                "Study_Satisfaction": 3.0,
                "Sleep_Duration": 1,
                "Dietary_Habits": 1,
                "Degree": 5,
                "Suicidal_Thoughts": 0,
                "Work_Study_Hours": 6.0,
                "Financial_Stress": 2.0,
                "Family_History": 0,
            }
        }
    }


# ── Output schemas ────────────────────────────────────────────────────────────
class PredictionOutput(BaseModel):
    depression_score: float = Field(
        description="Predicted depression risk score (0–1, higher = more risk)"
    )
    risk_label: str = Field(
        description="'Depressed' or 'Not Depressed'"
    )
    confidence: str = Field(
        description="High, Medium, or Low"
    )


class RetrainOutput(BaseModel):
    message: str
    new_mse: float
    new_r2: float
    rows_used: int


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "API is running",
        "model": "LinearRegression",
        "docs": "/docs",
        "predict_endpoint": "/predict",
        "retrain_endpoint": "/retrain",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model": "LinearRegression"}


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(data: StudentInput):
    """
    Predict depression risk for a single student.

    All categorical fields must be passed as their **encoded integer values**
    (matching what LabelEncoder produced during training). No re-encoding
    is done server-side — the scaler expects numeric input directly.

    - Score >= 0.5 → **Depressed**
    - Score < 0.5  → **Not Depressed**
    """
    try:
        # Build input array in EXACT same column order as training
        input_array = np.array([[
            data.Age,               # Age
            data.Gender,            # Gender (0/1)
            data.Academic_Pressure, # Academic Pressure
            data.Work_Pressure,     # Work Pressure
            data.CGPA,              # CGPA
            data.Study_Satisfaction,# Study Satisfaction
            data.Sleep_Duration,    # Sleep Duration (0-4)
            data.Dietary_Habits,    # Dietary Habits (0-2)
            data.Degree,            # Degree (0-20)
            data.Suicidal_Thoughts, # Suicidal Thoughts (0/1)
            data.Work_Study_Hours,  # Work/Study Hours
            data.Financial_Stress,  # Financial Stress
            data.Family_History,    # Family History (0/1)
        ]], dtype=float)

        # Scale using the same scaler fit during training
        input_scaled = scaler.transform(input_array)

        # Predict and clamp to [0, 1]
        raw_score = float(model.predict(input_scaled)[0])
        score = max(0.0, min(1.0, raw_score))

        risk_label = "Depressed" if score >= 0.5 else "Not Depressed"

        distance = abs(score - 0.5)
        if distance >= 0.35:
            confidence = "High"
        elif distance >= 0.15:
            confidence = "Medium"
        else:
            confidence = "Low"

        return PredictionOutput(
            depression_score=round(score, 4),
            risk_label=risk_label,
            confidence=confidence,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/retrain", response_model=RetrainOutput, tags=["Retraining"])
async def retrain(file: UploadFile = File(...)):
    """
    Retrain the model by uploading a new CSV file.

    The CSV must contain the same columns as the original training dataset.
    The model and scaler are updated in memory and saved to disk automatically.

    **Required columns:**
    Age, Gender, Academic Pressure, Work Pressure, CGPA,
    Study Satisfaction, Sleep Duration, Dietary Habits, Degree,
    Have you ever had suicidal thoughts ?, Work/Study Hours,
    Financial Stress, Family History of Mental Illness, Depression
    """
    global model, scaler

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted."
        )

    try:
        contents = await file.read()
        new_df = pd.read_csv(io.BytesIO(contents))

        required_cols = [
            "Age", "Gender", "Academic Pressure", "Work Pressure", "CGPA",
            "Study Satisfaction", "Sleep Duration", "Dietary Habits", "Degree",
            "Have you ever had suicidal thoughts ?", "Work/Study Hours",
            "Financial Stress", "Family History of Mental Illness", "Depression"
        ]
        missing = [c for c in required_cols if c not in new_df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )

        if len(new_df) < 50:
            raise HTTPException(
                status_code=400,
                detail="Need at least 50 rows to retrain."
            )

        feature_raw = [
            "Age", "Gender", "Academic Pressure", "Work Pressure", "CGPA",
            "Study Satisfaction", "Sleep Duration", "Dietary Habits", "Degree",
            "Have you ever had suicidal thoughts ?", "Work/Study Hours",
            "Financial Stress", "Family History of Mental Illness"
        ]

        X_new = new_df[feature_raw].copy()
        y_new = new_df["Depression"]

        # Split first — no leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X_new, y_new, test_size=0.2, random_state=42
        )

        # Impute Financial Stress using train median only
        train_median = X_train["Financial Stress"].median()
        X_train["Financial Stress"] = X_train["Financial Stress"].fillna(train_median)
        X_test["Financial Stress"] = X_test["Financial Stress"].fillna(train_median)

        # Encode categoricals on train only
        cat_cols = X_train.select_dtypes(include="object").columns.tolist()
        le = LabelEncoder()
        for col in cat_cols:
            le.fit(X_train[col].astype(str))
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = X_test[col].apply(
                lambda x: le.transform([str(x)])[0]
                if str(x) in le.classes_ else -1
            )

        # Scale on train only
        new_scaler = StandardScaler()
        X_train_scaled = new_scaler.fit_transform(X_train)
        X_test_scaled = new_scaler.transform(X_test)

        # Retrain
        new_model = LinearRegression()
        new_model.fit(X_train_scaled, y_train)

        preds = new_model.predict(X_test_scaled)
        new_mse = float(mean_squared_error(y_test, preds))
        new_r2 = float(r2_score(y_test, preds))

        # Save and hot-swap
        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(new_scaler, SCALER_PATH)
        model = new_model
        scaler = new_scaler

        return RetrainOutput(
            message="Model retrained and updated successfully.",
            new_mse=round(new_mse, 6),
            new_r2=round(new_r2, 4),
            rows_used=len(new_df),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("prediction:app", host="0.0.0.0", port=8000, reload=True)
