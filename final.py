import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import csv
import os
import requests
from io import StringIO

# Load the data from Google Drive
file_id = '1udpZDVUa_vnvHdKxTCKrNvh3ClIdv5as'  # Replace with your file ID from Google Drive
DOWNLOAD_URL = f'https://drive.google.com/uc?export=download&id={file_id}'

# Download and load the data
response = requests.get(DOWNLOAD_URL)
csv_data = StringIO(response.text)
data = pd.read_csv(csv_data).dropna(axis=1)

# Data preprocessing
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Initialize models
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)

# Train the models
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Define symptoms and classes
symptoms = X.columns.values
data_dict = {
    "symptom_index": {symptom: index for index, symptom in enumerate(symptoms)},
    "predictions_classes": encoder.classes_
}

# Ensure flagged data file exists
FLAGGED_DATA_PATH = "flagged_data.csv"
if not os.path.exists(FLAGGED_DATA_PATH):
    with open(FLAGGED_DATA_PATH, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["Symptom1", "Symptom2", "Symptom3", "RF_Prediction", "NB_Prediction", "SVM_Prediction", "Final_Prediction"])

# Prediction function
def predict_disease(symptom1, symptom2, symptom3, flag=False):
    input_symptoms = [symptom1, symptom2, symptom3]
    
    # Check if all symptoms are filled
    if any(symptom.strip() == '' for symptom in input_symptoms):
        return "Please fill in all three symptoms."
    
    # Create input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in input_symptoms:
        index = data_dict["symptom_index"].get(symptom.lower())
        if index is not None:
            input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    # Generate individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    
    # Make final prediction by taking mode of all predictions
    final_prediction = max(set([rf_prediction, nb_prediction, svm_prediction]), 
                           key=[rf_prediction, nb_prediction, svm_prediction].count)
    
    if flag:
        with open(FLAGGED_DATA_PATH, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([symptom1, symptom2, symptom3, rf_prediction, nb_prediction, svm_prediction, final_prediction])
    
    return {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }

# Streamlit app
st.title("Disease Predictor")
st.markdown("---")

# Sidebar for additional controls
with st.sidebar:
    st.title("Customize Interface")
    st.markdown("### Personalize your experience")

    # Example of a slider for customization
    st.markdown("#### Adjust Font Size")
    font_size = st.slider("Select Font Size", min_value=10, max_value=30, step=2, value=14)

# Main content area
st.header("Predict Disease")

# Create dropdowns for symptoms
with st.form("symptoms_form"):
    st.write("### Enter Symptoms")
    symptom1 = st.selectbox("Select Symptom 1", symptoms)
    symptom2 = st.selectbox("Select Symptom 2", symptoms)
    symptom3 = st.selectbox("Select Symptom 3", symptoms)

    # Create a checkbox for flagging the record
    flag = st.checkbox("Flag this record")

    # Prediction button
    predict_button = st.form_submit_button("Predict Disease")

# Display predictions
if predict_button:
    result = predict_disease(symptom1, symptom2, symptom3, flag)
    if isinstance(result, str):
        st.error(result)  # Show error message if not all symptoms are filled
    else:
        st.markdown(f"### Predicted Disease: **{result['final_prediction']}**")
        st.markdown(f"**RF Model Prediction:** {result['rf_model_prediction']}")
        st.markdown(f"**Naive Bayes Prediction:** {result['naive_bayes_prediction']}")
        st.markdown(f"**SVM Model Prediction:** {result['svm_model_prediction']}")

# Custom CSS to change the interface appearance
st.markdown(
    f"""
    <style>
        body {{
            font-size: {font_size}px;
            color: #333;
        }}
        .sidebar .sidebar-content {{
            background-color: #f0f0f0;
            padding: 20px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# To run the Streamlit app, use the following command in your terminal:
# streamlit run app.py
