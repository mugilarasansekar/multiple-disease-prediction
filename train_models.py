# Import Packages
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Evaluation Function
def evaluate(model_name, y_test, y_pred):
  print(f"{model_name} Evaluation:")
  print(f"Accuracy Score: {accuracy_score(y_test, y_pred)*100:.2f}%")
  print(f"Precision Score: {precision_score(y_test, y_pred)*100:.2f}%")
  print(f"Recall Score: {recall_score(y_test, y_pred)*100:.2f}%")
  print(f"F1 Score: {f1_score(y_test, y_pred)*100:.2f}%")
  print("Confusion Matrix:")
  print(confusion_matrix(y_test, y_pred))
  print()

# -------------------------
# Parkinsons Disease
df_p = pd.read_csv("./datasets/parkinsons.csv")

# Feature and Target
X_p = df_p.drop(columns=["name", "status"])
y_p = df_p["status"]

# Feature names for Streamlit
feature_names_p = X_p.columns.tolist()

# Train Test Split
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.2, random_state=42, stratify=y_p)

# Scaling
scaler_p = StandardScaler()
X_train_scaled_p = scaler_p.fit_transform(X_train_p)
X_test_scaled_p = scaler_p.transform(X_test_p)

# Model Training
# Logistic Regression
lr_p = LogisticRegression(max_iter=1000, random_state=42)
lr_p.fit(X_train_scaled_p, y_train_p)
lr_pred_p = lr_p.predict(X_test_scaled_p)
evaluate("Parkinsons - Logistic Regression", y_test_p, lr_pred_p)

# Random forest
rf_p = RandomForestClassifier(random_state=42)
rf_p.fit(X_train_scaled_p, y_train_p)
rf_pred_p = rf_p.predict(X_test_scaled_p)
evaluate("Parkinsons - Random Forest", y_test_p, rf_pred_p)

# XGBoost
xg_p = XGBClassifier(eval_metric="logloss", random_state=42)
xg_p.fit(X_train_scaled_p, y_train_p)
xg_pred_p = xg_p.predict(X_test_scaled_p)
evaluate("Parkinsons - XGBoost", y_test_p, xg_pred_p)

# --------------------------
# Liver Disease
df_l = pd.read_csv("./datasets/indian_liver_patient.csv")

# mapping "Dataset" 1(liver disease) as 1 and 2(no disease) as 0
df_l["Dataset"] = df_l["Dataset"].map({
    1: 1,
    2: 0
})

# Encode Gender
df_l["Gender"] = df_l["Gender"].map({
    "Male": 1,
    "Female": 0
})

# Feature and Target
X_l = df_l.drop(columns=["Dataset"])
y_l = df_l["Dataset"]

# Train Test Split
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l, y_l, test_size=0.2, random_state=42, stratify=y_l)

# Filling null values with median
X_train_l["Albumin_and_Globulin_Ratio"] = X_train_l["Albumin_and_Globulin_Ratio"].fillna(X_train_l["Albumin_and_Globulin_Ratio"].median())
X_test_l["Albumin_and_Globulin_Ratio"] = X_test_l["Albumin_and_Globulin_Ratio"].fillna(X_train_l["Albumin_and_Globulin_Ratio"].median())

# Outlier features
outlier_cols = ["Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase"]

# Using Log Transformation to compress the outlier
for col in outlier_cols:
    X_train_l[col+"log"] = np.log1p(X_train_l[col])
    X_test_l[col+"log"] = np.log1p(X_test_l[col])

# Dropping old outlier columns
X_train_l = X_train_l.drop(columns=outlier_cols)
X_test_l = X_test_l.drop(columns=outlier_cols)

# Dropping Direct_Bilirubinlog
X_train_l = X_train_l.drop(columns=["Direct_Bilirubinlog"])
X_test_l = X_test_l.drop(columns=["Direct_Bilirubinlog"])

# Feature names
feature_names_l = X_train_l.columns.tolist()

# Scaling
scaler_l = StandardScaler()
X_train_scaled_l = scaler_l.fit_transform(X_train_l)
X_test_scaled_l = scaler_l.transform(X_test_l)

# Model Training
# Logistic Regression
lr_l = LogisticRegression(max_iter=1000, random_state=42)
lr_l.fit(X_train_scaled_l, y_train_l)
lr_pred_l = lr_l.predict(X_test_scaled_l)
evaluate("Liver - Logistic Regression", y_test_l, lr_pred_l)

# Random forest
rf_l = RandomForestClassifier(random_state=42)
rf_l.fit(X_train_scaled_l, y_train_l)
rf_pred_l = rf_l.predict(X_test_scaled_l)
evaluate("Liver - Random Forest", y_test_l, rf_pred_l)

# XGBoost
xg_l = XGBClassifier(eval_metric="logloss", random_state=42)
xg_l.fit(X_train_scaled_l, y_train_l)
xg_pred_l = xg_l.predict(X_test_scaled_l)
evaluate("Liver - XGBoost", y_test_l, xg_pred_l)

# -----------------
# Kidney Disease
df_k = pd.read_csv("./datasets/kidney_disease.csv")

# Dropping id columns
df_k = df_k.drop(columns=["id"])

# converting ? in pcv, wc, rc to null value
df_k["pcv"] = pd.to_numeric(df_k["pcv"], errors="coerce")
df_k["wc"] = pd.to_numeric(df_k["wc"], errors="coerce")
df_k["rc"] = pd.to_numeric(df_k["rc"], errors="coerce")

# Encode classification column
df_k["classification"] = df_k["classification"].map({
    "ckd": 1,
    "notckd": 0
})

num_cols = df_k.select_dtypes(include=["float64"]).columns.tolist()
cat_cols = df_k.select_dtypes(include=["object"]).columns.tolist()

# Encoding cat_cols
values = {
    "normal": 1, "abnormal": 0,
    "present": 1, "notpresent": 0,
    "yes": 1, "no": 0,
    "good": 1, "poor": 0
}
for i in cat_cols:
  df_k[i] = df_k[i].map(values)

# Feature and Target
X_k = df_k.drop(columns=["classification"])
y_k = df_k["classification"]

# Train Test Split
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X_k, y_k, test_size=0.2, random_state=42, stratify=y_k)

# Fill missing values
for i in num_cols:
    median_value = X_train_k[i].median()
    X_train_k[i] = X_train_k[i].fillna(median_value)
    X_test_k[i] = X_test_k[i].fillna(median_value)

for i in cat_cols:
    mode_value = X_train_k[i].mode()[0]
    X_train_k[i] = X_train_k[i].fillna(mode_value)
    X_test_k[i] = X_test_k[i].fillna(mode_value)

# Dropping pcv
X_train_k = X_train_k.drop(columns=["pcv"])
X_test_k = X_test_k.drop(columns=["pcv"])

# Log Transform highly skewed outliers
highly_skewed_outliers = ["pot", "sc", "bu", "bgr", "wc", "bp"]

for i in highly_skewed_outliers:
    X_train_k[i+"_log"] = np.log1p(X_train_k[i])
    X_test_k[i+"_log"] = np.log1p(X_test_k[i])

# dropping old outlier columns
X_train_k = X_train_k.drop(columns=highly_skewed_outliers)
X_test_k = X_test_k.drop(columns=highly_skewed_outliers)

# Feature names
feature_names_k = X_train_k.columns.tolist()

# Scaling
scaler_k = StandardScaler()
X_train_scaled_k = scaler_k.fit_transform(X_train_k)
X_test_scaled_k = scaler_k.transform(X_test_k)

# Model Training
# Logistic Regression
lr_k = LogisticRegression(max_iter=1000, random_state=42)
lr_k.fit(X_train_scaled_k, y_train_k)
lr_pred_k = lr_k.predict(X_test_scaled_k)
evaluate("Kidney - Logistic Regression", y_test_k, lr_pred_k)

# Random forest
rf_k = RandomForestClassifier(random_state=42)
rf_k.fit(X_train_scaled_k, y_train_k)
rf_pred_k = rf_k.predict(X_test_scaled_k)
evaluate("Kidney - Random Forest", y_test_k, rf_pred_k)

# XGBoost
xg_k = XGBClassifier(eval_metric="logloss", random_state=42)
xg_k.fit(X_train_scaled_k, y_train_k)
xg_pred_k = xg_k.predict(X_test_scaled_k)
evaluate("Kidney - XGBoost", y_test_k, xg_pred_k)

# Save model files
models = {
    "parkinsons_lr": lr_p, "parkinsons_rf": rf_p, "parkinsons_xg": xg_p,
    "liver_lr": lr_l, "liver_rf": rf_l, "liver_xg": xg_l,
    "kidney_lr": lr_k, "kidney_rf": rf_k, "kidney_xg": xg_k
}
scalers = {
    "parkinsons": scaler_p,
    "liver": scaler_l,
    "kidney": scaler_k
}

path = "./models/"
for i, j in models.items():
  joblib.dump(j, path+i+"_model.pkl")
for i, j in scalers.items():
  joblib.dump(j, path+i+"_scaler.pkl")

# Save feature names
joblib.dump(feature_names_p, path+"parkinsons_features.pkl")
joblib.dump(feature_names_l, path+"liver_features.pkl")
joblib.dump(feature_names_k, path+"kidney_features.pkl")