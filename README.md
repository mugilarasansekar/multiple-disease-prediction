# Multiple Disease Prediction

## Problem
Early detection of disease plays a major role in reducing treatment cost and improving patient outcomes. Manual diagnosis based on lab reports takes time and depends on specialist availability. This project aims to use patient test data to predict the likelihood of three diseases, Parkinsons, Liver Disease, and Chronic Kidney Disease (CKD), and make these predictions accessible through a single Streamlit app.

## Predictions
1. **Parkinson's Disease** — does the patient show signs of Parkinson's, based on voice measurement data
2. **Liver Disease** — does the patient show signs of liver disease, based on blood test values
3. **Chronic Kidney Disease** — does the patient show signs of CKD, based on blood and urine test values

## Process
- Built and evaluated all three sections in a single Colab notebook, one disease at a time (Parkinson's → Liver → Kidney)
- Used a variable naming convention with dataset suffixes (`_p`, `_l`, `_k`) throughout, so variables from one section can't accidentally leak into another
- For each disease: handled nulls, encoded categorical columns, treated outliers/skewness, checked correlation, split the data, scaled, trained, evaluated, saved
- **Split before impute** — for Kidney, missing values were originally filled before the train/test split, which is data leakage. Fixed by fitting median/mode only on `X_train`, then applying that same value to `X_test`
- Applied `np.log1p()` log transformation on skewed columns (skew > 0.5) instead of removing outliers, since extreme values in medical data (e.g. bilirubin, creatinine) often carry real clinical signal
- Dropped one feature from highly correlated pairs (correlation ≥ ~0.85), kept moderately correlated pairs (0.78–0.84) since they can still carry independent signal
- Trained 3 models per disease — Logistic Regression, Random Forest, XGBoost — and compared using Accuracy, Precision, Recall, F1, Confusion Matrix, and 5-fold Cross-Validation F1 (used as the tiebreaker when test-set metrics were close, especially on the small Liver dataset)
- Built a reusable `evaluate()` function instead of repeating evaluation code for every model
- For Liver, tuned the classification threshold on Logistic Regression — 0.45 gave the most balanced recall across classes (default 0.5 missed too many positive cases)

## Final Model Selection
| Disease | Model | Why |
|---|---|---|
| Parkinson's | XGBoost | Best mean CV F1 score |
| Liver | Logistic Regression (threshold 0.45) | Best macro F1, most balanced recall on a small dataset — simpler model generalized better than RF/XGBoost |
| Kidney (CKD) | Random Forest | Best CV F1; near-perfect scores confirmed genuine via CV, since the UCI CKD dataset is known to be linearly separable |

## Key Notes
- The Liver model needs `model.predict_proba()[:, 1] >= 0.45` in the app, not the default `.predict()` — using `.predict()` would silently apply the wrong threshold
- With small test sets, different models can land on identical point-in-time metrics — cross-validation is what actually separates them
- Colab keeps variables in memory between cells, so reusing a variable name (e.g. `y_test` instead of `y_test_p`) can silently pull in stale data from an earlier section — the suffix convention exists to prevent this

## Files
- `Multiple_Disease_Prediction.ipynb` — Data Loading, Cleaning, Preprocessing, EDA, Model Building, Evaluation, Pickle (all three diseases)
- `train_models.py` — script version of the notebook's modeling pipeline, used to generate the model files
- `streamlit_report.py` — Streamlit app, loads model files via joblib and serves predictions for all three diseases
- `parkinsons.csv`, `indian_liver_patient.csv`, `kidney_disease.csv` — raw datasets
- `models/` — folder containing all saved `.pkl` files:
  - `parkinsons_lr_model.pkl`, `parkinsons_rf_model.pkl`, `parkinsons_xg_model.pkl`, `parkinsons_scaler.pkl`, `parkinsons_features.pkl`
  - `liver_lr_model.pkl`, `liver_rf_model.pkl`, `liver_xg_model.pkl`, `liver_scaler.pkl`, `liver_features.pkl`
  - `kidney_lr_model.pkl`, `kidney_rf_model.pkl`, `kidney_xg_model.pkl`, `kidney_scaler.pkl`, `kidney_features.pkl`

## How to Run
1. `pip install -r requirements.txt`
2. Place `parkinsons.csv`, `indian_liver_patient.csv`, `kidney_disease.csv` inside a `datasets/` folder
3. `python train_models.py` — trains all models and saves `.pkl` files into `models/`
4. `streamlit run streamlit_report.py`

## App Modes
- **Basic Mode** — pick a disease, fill the form, get a prediction from the best model for that disease
- **Advanced Mode** — pick a disease and manually choose which model (Logistic Regression, Random Forest, or XGBoost) to predict with

## Deployment
Streamlit link: https://multiple-disease-prediction-data.streamlit.app/