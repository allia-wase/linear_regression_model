# Predicting Depression Risk Among University Students

This project predicts depression risk in university students using machine learning to enable early identification and timely mental health intervention. The model uses student lifestyle and academic factors such as academic pressure, financial stress, sleep duration, and CGPA to predict depression risk (0–1 score; higher = higher risk).

---

## Dataset

| | |
|---|---|
| **Source** | [Student Depression Dataset — Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) |
| **Size** | 27,901 rows × 18 columns |
| **Target variable** | Depression (binary: 0 = Not Depressed, 1 = Depressed) |
| **Key features** | Age, CGPA, Academic Pressure, Financial Stress, Sleep Duration, Dietary Habits, Suicidal Thoughts, Family History of Mental Illness |

---

## API Endpoint

**Public URL:** `https://mindease-n866.onrender.com` (replace with your deployed Render URL)

**Swagger UI:** `https://mindease-n866.onrender.com/docs`

**Prediction endpoint:** `POST /predict`

---

## Repository Structure

```
linear_regression_model/
│
├── Summative/
│   ├── linear_regression/
│   │   └── Multivariate.ipynb
│   ├── API/
│   │   ├── main.py
│   │   ├── prediction.py
│   │   ├── requirements.txt
│   │   └── render.yaml
│   └── FlutterApp/
```

---

## How to Run the Mobile App

1. Install Flutter: https://docs.flutter.dev/get-started/install
2. Open the Flutter app: `cd Summative/FlutterApp`
3. Install dependencies: `flutter pub get`
4. Update the API URL in `lib/main.dart`: set `apiBaseUrl` to your deployed API URL (e.g. `https://your-api.onrender.com`)
5. Run on device or emulator: `flutter run`

---

## Video Demo
https://drive.google.com/file/d/1cr3BLNRhvpBGUHyqQpyh7WXhKIWe89F5/view?usp=sharing
