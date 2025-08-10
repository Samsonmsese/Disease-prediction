import streamlit as st
import numpy as np
import pandas as pd
from statistics import mode
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# 🔧 Load and preprocess data
@st.cache_data
def load_data():
    # Load dataset
    data = pd.read_csv('improved_disease_dataset.csv')

    # Encode target labels
    encoder = LabelEncoder()
    data["disease"] = encoder.fit_transform(data["disease"])

    # Separate features and target
    X = data.drop("disease", axis=1)
    y = data["disease"]

    # Handle missing values
    X = X.fillna(0)

    # Encode all object-type columns
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # ✅ Ensure X is a DataFrame and y is a Series
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # ✅ Check shapes before resampling
    if len(X.shape) != 2 or X.shape[0] != len(y):
        raise ValueError(f"Invalid shapes: X={X.shape}, y={y.shape}")

    # Apply RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    return X_resampled, y_resampled, encoder, X.columns.values

# 🚀 Train models
@st.cache_resource
def train_models(X, y):
    rf_model = RandomForestClassifier()
    rf_model.fit(X, y)

    nb_model = GaussianNB()
    nb_model.fit(X, y)

    svm_model = SVC()
    svm_model.fit(X, y)

    return rf_model, nb_model, svm_model

# 🔍 Prediction function
def predict_disease(symptom_input, rf_model, nb_model, svm_model, encoder, symptom_index):
    input_data = [0] * len(symptom_index)
    for symptom in symptom_input:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1
    input_data = np.array(input_data).reshape(1, -1)

    rf_pred = encoder.classes_[rf_model.predict(input_data)[0]]
    nb_pred = encoder.classes_[nb_model.predict(input_data)[0]]
    svm_pred = encoder.classes_[svm_model.predict(input_data)[0]]
    final_pred = mode([rf_pred, nb_pred, svm_pred])

    return rf_pred, nb_pred, svm_pred, final_pred

# 🌐 Streamlit UI
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🧠 Disease Prediction App")
st.write("Enter symptoms below to predict the most likely disease.")

# Load data and models
X_resampled, y_resampled, encoder, symptoms = load_data()
rf_model, nb_model, svm_model = train_models(X_resampled, y_resampled)
symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)}

# Input field
user_input = st.text_input("Enter symptoms (comma-separated):", "Itching,Skin Rash,Nodal Skin Eruptions")

# Predict button
if st.button("Predict"):
    symptom_list = [s.strip() for s in user_input.split(",")]
    rf_pred, nb_pred, svm_pred, final_pred = predict_disease(symptom_list, rf_model, nb_model, svm_model, encoder, symptom_index)

    st.subheader("🔍 Predictions")
    st.write(f"**Random Forest:** {rf_pred}")
    st.write(f"**Naive Bayes:** {nb_pred}")
    st.write(f"**SVM:** {svm_pred}")
    st.success(f"**Final Ensemble Prediction:** ✅ {final_pred}")

