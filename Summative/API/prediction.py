"""
Prediction script for depression risk model.
Loads the best saved model and makes predictions given input features.
Used by the FastAPI endpoint and can be run standalone.
"""
import os
import joblib
import pandas as pd

# Path to artifacts (same directory as this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(SCRIPT_DIR, "encoders.pkl")
FEATURE_COLS_PATH = os.path.join(SCRIPT_DIR, "feature_cols.pkl")


def load_artifacts():
    """Load model, scaler, encoders, and feature columns."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model, scaler, encoders, feature_cols


def predict(features: dict) -> float:
    """
    Make a prediction given a dict of feature values (human-readable).
    
    Args:
        features: Dict with keys matching feature_cols. Categorical values
                  as strings (e.g. "Male", "Female", "5-6 hours").
    
    Returns:
        Predicted depression score (float, typically 0-1).
    """
    model, scaler, encoders, feature_cols = load_artifacts()
    
    # Build row in correct order
    row = {}
    for col in feature_cols:
        val = features.get(col)
        if val is None:
            raise ValueError(f"Missing required feature: {col}")
        row[col] = val
    
    df = pd.DataFrame([row])
    
    # Encode categorical columns
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Scale and predict
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return float(pred)


if __name__ == "__main__":
    # Example: predict for one student (values from test set)
    sample = {
        "Age": 29.0,
        "Gender": "Male",
        "Academic Pressure": 2.0,
        "Work Pressure": 0.0,
        "CGPA": 7.5,
        "Study Satisfaction": 4.0,
        "Sleep Duration": "5-6 hours",
        "Dietary Habits": "Healthy",
        "Degree": "BSc",
        "Have you ever had suicidal thoughts ?": "No",
        "Work/Study Hours": 6.0,
        "Financial Stress": 2.0,
        "Family History of Mental Illness": "No",
    }
    result = predict(sample)
    print(f"Predicted depression score: {result:.4f}")
