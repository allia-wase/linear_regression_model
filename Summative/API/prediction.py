import os
import io
import joblib
import numpy as np
import pandas as pd
import uvicorn  # pyright: ignore[reportMissingImports]

from fastapi import FastAPI, Request, UploadFile, File, HTTPException  # pyright: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Any, Dict, Literal, Optional

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
        "https://mindease-n866.onrender.com",
        "https://your-app.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# ── Load model and scaler ─────────────────────────────────────────────────────
# All paths are relative to wherever prediction.py lives on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

FEATURE_COLS = [
    "Age",
    "Gender",
    "Academic Pressure",
    "Work Pressure",
    "CGPA",
    "Study Satisfaction",
    "Sleep Duration",
    "Dietary Habits",
    "Degree",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours",
    "Financial Stress",
    "Family History of Mental Illness",
]

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"best_model.pkl not found at {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise RuntimeError(f"scaler.pkl not found at {SCALER_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

_encoders: Optional[Dict[str, LabelEncoder]] = None
if os.path.exists(ENCODERS_PATH):
    _encoders = joblib.load(ENCODERS_PATH)

# Flutter uses POST /predict with plain-language fields; encoders.pkl maps them
# to integers (same as notebook LabelEncoder). File is created by POST /retrain.

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
        ..., ge=0, le=3,
        description="Label-encoded dietary habit (0–3 per training data)"
    )
    Degree: int = Field(
        ..., ge=0, le=50,
        description="Label-encoded degree (integer index from training)"
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


class FlutterFormInput(BaseModel):
    """JSON body from the Flutter app (snake_case, human-readable categories)."""

    age: float = Field(..., ge=15.0, le=60.0)
    gender: Literal["Male", "Female"]
    academic_pressure: float = Field(..., ge=0.0, le=5.0)
    work_pressure: float = Field(..., ge=0.0, le=5.0)
    cgpa: float = Field(..., ge=0.0, le=10.0)
    study_satisfaction: float = Field(..., ge=0.0, le=5.0)
    sleep_duration: Literal[
        "5-6 hours",
        "7-8 hours",
        "Less than 5 hours",
        "More than 8 hours",
        "Others",
    ]
    dietary_habits: Literal["Healthy", "Moderate", "Unhealthy", "Others"]
    degree: str = Field(..., min_length=1, max_length=80)
    suicidal_thoughts: Literal["Yes", "No"]
    work_study_hours: float = Field(..., ge=0.0, le=24.0)
    financial_stress: float = Field(..., ge=1.0, le=5.0)
    family_history_mental_illness: Literal["Yes", "No"]


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


def _row_from_flutter(data: FlutterFormInput) -> Dict[str, Any]:
    return {
        "Age": data.age,
        "Gender": data.gender,
        "Academic Pressure": data.academic_pressure,
        "Work Pressure": data.work_pressure,
        "CGPA": data.cgpa,
        "Study Satisfaction": data.study_satisfaction,
        "Sleep Duration": data.sleep_duration,
        "Dietary Habits": data.dietary_habits,
        "Degree": data.degree.strip(),
        "Have you ever had suicidal thoughts ?": data.suicidal_thoughts,
        "Work/Study Hours": data.work_study_hours,
        "Financial Stress": data.financial_stress,
        "Family History of Mental Illness": data.family_history_mental_illness,
    }


def _encode_row_with_encoders(row: Dict[str, Any], encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    X = pd.DataFrame([row])
    for col, le in encoders.items():
        if col not in X.columns:
            continue
        val = str(X[col].iloc[0])
        if val not in le.classes_:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid value for {col!r}: {val!r}. "
                    f"Allowed labels: {list(le.classes_)}"
                ),
            )
        X[col] = le.transform([val])[0]
    return X


def _prediction_output_from_scaled(input_scaled: np.ndarray) -> "PredictionOutput":
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


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "API is running",
        "model": "LinearRegression",
        "docs": "/docs",
        "predict_flutter": "/predict",
        "predict_encoded": "/predict/encoded",
        "retrain_endpoint": "/retrain",
        "note": "POST /predict accepts Flutter JSON (age, gender, …) or encoded JSON (Age, Gender, …). "
        "Flutter path needs encoders.pkl from POST /retrain.",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model": "LinearRegression"}


def _predict_flutter_impl(data: FlutterFormInput) -> PredictionOutput:
    global _encoders
    if _encoders is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "encoders.pkl is missing. Call POST /retrain once with your training CSV, "
                "then try again."
            ),
        )
    row = _row_from_flutter(data)
    X = _encode_row_with_encoders(row, _encoders)
    X_ord = X[FEATURE_COLS].astype(float)
    input_scaled = scaler.transform(X_ord)
    return _prediction_output_from_scaled(input_scaled)


def _predict_student_impl(data: StudentInput) -> PredictionOutput:
    input_array = np.array(
        [[
            data.Age,
            data.Gender,
            data.Academic_Pressure,
            data.Work_Pressure,
            data.CGPA,
            data.Study_Satisfaction,
            data.Sleep_Duration,
            data.Dietary_Habits,
            data.Degree,
            data.Suicidal_Thoughts,
            data.Work_Study_Hours,
            data.Financial_Stress,
            data.Family_History,
        ]],
        dtype=float,
    )
    input_scaled = scaler.transform(input_array)
    return _prediction_output_from_scaled(input_scaled)


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(request: Request):
    """
    **Two JSON shapes** (auto-detected):

    - **Flutter:** snake_case keys (`age`, `gender`, `academic_pressure`, …) and plain-language
      categories. Needs **encoders.pkl** (from **POST /retrain**).
    - **Encoded / Swagger example:** PascalCase keys (`Age`, `Gender`, `Academic_Pressure`, …)
      with integer-encoded categories — same as the notebook export.
    """
    try:
        raw = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="JSON object required")

    if "Age" in raw:
        try:
            data = StudentInput.model_validate(raw)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        try:
            return _predict_student_impl(data)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    try:
        data = FlutterFormInput.model_validate(raw)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    try:
        return _predict_flutter_impl(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/encoded", response_model=PredictionOutput, tags=["Prediction"])
def predict_encoded(data: StudentInput):
    """Same as **POST /predict** with PascalCase `Age` body (alias for tools/tests)."""
    try:
        return _predict_student_impl(data)
    except HTTPException:
        raise
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
    global model, scaler, _encoders

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

        # Encode categoricals on train only; persist encoders for POST /predict
        cat_cols = X_train.select_dtypes(include="object").columns.tolist()
        encoders_dict: Dict[str, LabelEncoder] = {}
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(X_train[col].astype(str))
            encoders_dict[col] = le
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = X_test[col].apply(
                lambda x, enc=le: enc.transform([str(x)])[0]
                if str(x) in enc.classes_ else -1
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
        joblib.dump(encoders_dict, ENCODERS_PATH)
        model = new_model
        scaler = new_scaler
        _encoders = encoders_dict

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
