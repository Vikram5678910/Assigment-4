import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Sidebar: disease selection ---
st.sidebar.title("Select Disease")
disease_option = st.sidebar.radio("Choose a disease:", ["Parkinson's", "Kidney", "Liver"])

# --- Load models and features ---
base_path = r"C:\Users\gold\Documents\python\project-4\models"

def load_model(model_file, scaler_file, features_file):
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    features = joblib.load(features_file)
    return model, scaler, features

# --- Parkinson's ---
if disease_option == "Parkinson's":
    model, scaler, features = load_model(
        f"{base_path}/parkinson_lr.pkl",
        f"{base_path}/scaler_parkinsons.pkl",
        f"{base_path}/parkinson_features.pkl"
    )

    st.title("Parkinson's Disease Prediction ")
   

    input_data = {}
    for feat in features:
        
        input_data[feat] = st.number_input(feat, min_value=0.0, max_value=10.0, step=0.00001, format="%.5f", key=feat)

    if st.button("Predict"):
        df_input = pd.DataFrame([input_data])
        df_scaled = scaler.transform(df_input)
        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1]

        
        if prob < 0.3:
            color = "green"
            emoji = " Healthy"
        elif prob < 0.6:
            color = "yellow"
            emoji = " Mild Risk"
        else:
            color = "red"
            emoji = "High Risk"

        st.markdown(f"**Prediction:** {emoji}")
        st.progress(int(prob * 100))
        st.write(f"Probability of Parkinson's: {prob:.2f}")

# --- Kidney ---
elif disease_option == "Kidney":
    model, scaler, features = load_model(
        f"{base_path}/Kidney_rf.pkl",
        f"{base_path}/scaler_Kidney.pkl",
        f"{base_path}/Kidney_features.pkl"
    )

    st.title("Kidney Disease Prediction ")
   

    input_data = {}
    for feat in features:
        input_data[feat] = st.number_input(feat, min_value=0.0, max_value=100.0, step=0.01, format="%.5f", key=feat)

    if st.button("Predict"):
        df_input = pd.DataFrame([input_data])
        df_scaled = scaler.transform(df_input)
        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1]

        if prob < 0.3:
            color = "green"
            emoji = " Healthy"
        elif prob < 0.7:
            color = "yellow"
            emoji = " Mild Risk"
        else:
            color = "red"
            emoji = " High Risk"

        st.markdown(f"**Prediction:** {emoji}")
        st.progress(int(prob * 100))
        st.write(f"Probability of CKD: {prob:.2f}")

# --- Load Liver model ---
model_path = r"C:\Users\gold\Documents\python\project-4\models\liver_rf.pkl"
scaler_path = r"C:\Users\gold\Documents\python\project-4\models\scaler_liver.pkl"
features_path = r"C:\Users\gold\Documents\python\project-4\models\liver_features.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(features_path)  # this must already include "gender"

st.title("Liver Disease Prediction 🫁")

# --- Special case: Gender input ---
gender = st.radio("Gender:", ["Male", "Female"], horizontal=True)
gender_value = 1 if gender == "Male" else 0

# --- Manual inputs for other features ---
input_data = {"gender": gender_value}
for feat in feature_names:
    if feat != "gender":  # avoid duplicate gender
        input_data[feat] = st.number_input(feat, step=0.01, format="%.2f")

# --- Prediction ---
if st.button("Predict"):
    df_input = pd.DataFrame([input_data], columns=feature_names)  # keep exact feature order
    df_scaled = scaler.transform(df_input)

    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    # --- Risk visualization ---
    if prob < 0.3:
        emoji, color = "🟢 Healthy", "green"
    elif prob < 0.6:
        emoji, color = "🟡 Mild Risk", "yellow"
    else:
        emoji, color = "🔴 High Risk", "red"

    st.markdown(f"### Prediction: {emoji}")
    st.progress(int(prob * 100))
    st.write(f"Probability of Liver Disease: **{prob*100:.2f}%**")
    
    st.subheader("Patient Input Values")
    plt.figure(figsize=(10,5))
    plt.bar(df_input.columns, df_input.iloc[0], color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel("Value")
    plt.title("Patient Liver Feature Inputs")
    st.pyplot(plt)

 