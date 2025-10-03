import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Base paths
base_path = r"C:\Users\gold\Documents\python\project-4"
model_path = os.path.join(base_path, "models")
os.makedirs(model_path, exist_ok=True)

paths = {
    'parkinsons': os.path.join(base_path, "parkinsons.xlsx"),
    'kidney': os.path.join(base_path, "kidney_disease.xlsx"),
    'liver': os.path.join(base_path, "indian_liver_patient.xlsx")
}

# Load datasets
def load_data(path):
    df = pd.read_excel(path)
    return df

parkinsons = load_data(paths['parkinsons']).drop(columns=['name'])
kidney = load_data(paths['kidney'])
liver = load_data(paths['liver']).drop_duplicates()

# ---------- Parkinson's ----------
X_par = parkinsons.drop(columns=['status'])
y_par = parkinsons['status']

Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_par, y_par, test_size=0.2, random_state=42, stratify=y_par)

scaler_par = StandardScaler()
Xp_train_scaled = pd.DataFrame(scaler_par.fit_transform(Xp_train), columns=Xp_train.columns)
Xp_test_scaled  = pd.DataFrame(scaler_par.transform(Xp_test), columns=Xp_test.columns)

lr_par = LogisticRegression(max_iter=500, random_state=42)
rf_par = RandomForestClassifier(n_estimators=100, random_state=42)

lr_par.fit(Xp_train_scaled, yp_train)
rf_par.fit(Xp_train_scaled, yp_train)

# Save Parkinson's models and scaler
joblib.dump(lr_par, os.path.join(model_path, "parkinson_lr.pkl"))
joblib.dump(rf_par, os.path.join(model_path, "parkinson_rf.pkl"))
joblib.dump(scaler_par, os.path.join(model_path, "scaler_parkinsons.pkl"))
joblib.dump(list(Xp_train_scaled.columns), os.path.join(model_path, "parkinson_features.pkl"))

# ---------- Kidney ----------
# Clean target
kidney['classification'] = kidney['classification'].astype(str).str.strip().str.lower()
kidney['classification'] = kidney['classification'].map({'ckd':1, 'notckd':0})
kidney = kidney.dropna(subset=['classification'])
kidney['classification'] = kidney['classification'].astype(int)

# Encode categorical
cat_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
for col in cat_cols:
    if col in kidney.columns:
        kidney[col] = kidney[col].fillna(kidney[col].mode()[0])
        le = LabelEncoder()
        kidney[col] = le.fit_transform(kidney[col].astype(str))

# Convert numeric-like columns
for col in kidney.columns:
    if kidney[col].dtype == object:
        kidney[col] = pd.to_numeric(kidney[col], errors='coerce')

# Fill numeric missing
for col in kidney.select_dtypes(include=['float64','int64']).columns:
    kidney[col] = kidney[col].fillna(kidney[col].mean())

# Split
X_kid = kidney.drop(columns=['classification','id'], errors='ignore')
y_kid = kidney['classification']
Xk_train, Xk_test, yk_train, yk_test = train_test_split(X_kid, y_kid, test_size=0.2, random_state=42, stratify=y_kid)

scaler_kid = StandardScaler()
Xk_train_scaled = pd.DataFrame(scaler_kid.fit_transform(Xk_train), columns=Xk_train.columns)
Xk_test_scaled  = pd.DataFrame(scaler_kid.transform(Xk_test), columns=Xk_test.columns)

lr_kid = LogisticRegression(max_iter=500, random_state=42)
rf_kid = RandomForestClassifier(n_estimators=100, random_state=42)

lr_kid.fit(Xk_train_scaled, yk_train)
rf_kid.fit(Xk_train_scaled, yk_train)

# Save Kidney models and scaler
joblib.dump(lr_kid, os.path.join(model_path, "kidney_lr.pkl"))
joblib.dump(rf_kid, os.path.join(model_path, "kidney_rf.pkl"))
joblib.dump(scaler_kid, os.path.join(model_path, "scaler_kidney.pkl"))
joblib.dump(list(Xk_train_scaled.columns), os.path.join(model_path, "kidney_features.pkl"))

# ---------- Liver ----------
liver['Albumin_and_Globulin_Ratio'] = liver['Albumin_and_Globulin_Ratio'].fillna(liver['Albumin_and_Globulin_Ratio'].mean())
liver['Gender'] = liver['Gender'].map({'Male':1, 'Female':0})
liver['Dataset'] = liver['Dataset'].map({1:1, 2:0})

X_liv = liver.drop(columns=['Dataset'])
y_liv = liver['Dataset']

Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_liv, y_liv, test_size=0.2, random_state=42, stratify=y_liv)

scaler_liv = StandardScaler()
Xl_train_scaled = pd.DataFrame(scaler_liv.fit_transform(Xl_train), columns=Xl_train.columns)
Xl_test_scaled  = pd.DataFrame(scaler_liv.transform(Xl_test), columns=Xl_test.columns)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(Xl_train_scaled, yl_train)

lr_liv = LogisticRegression(max_iter=500, random_state=42)
rf_liv = RandomForestClassifier(n_estimators=100, random_state=42)

lr_liv.fit(X_res, y_res)
rf_liv.fit(X_res, y_res)

# Save Liver models and scaler
joblib.dump(lr_liv, os.path.join(model_path, "liver_lr.pkl"))
joblib.dump(rf_liv, os.path.join(model_path, "liver_rf.pkl"))
joblib.dump(scaler_liv, os.path.join(model_path, "scaler_liver.pkl"))
joblib.dump(list(Xl_train_scaled.columns), os.path.join(model_path, "liver_features.pkl"))

print(f"\n✅ All models, scalers, and feature lists saved in folder: {model_path}")