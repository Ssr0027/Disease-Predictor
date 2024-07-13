import gradio as gr
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Load the data
DATA_PATH = "/Users/shubham/Desktop/DiseasePredictor/Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

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
symptoms = sorted(X.columns.values)  # Sort symptoms alphabetically
data_dict = {
    "symptom_index": {symptom: index for index, symptom in enumerate(symptoms)},
    "predictions_classes": encoder.classes_
}

# Prediction function
def predict_disease(symptom1, symptom2, symptom3):
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
    
    return {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }

# Gradio interface function with custom HTML and CSS
def interface(symptom1, symptom2, symptom3):
    prediction = predict_disease(symptom1, symptom2, symptom3)
    if isinstance(prediction, str):
        return prediction  # Return error message if not all symptoms are filled
    else:
        return f"""<h2><i class="fa-solid fa-user-doctor"></i> Predicted Disease: {prediction['final_prediction']}</h2>
                   <p>RF Model Prediction: {prediction['rf_model_prediction']}</p>
                   <p>Naive Bayes Prediction: {prediction['naive_bayes_prediction']}</p>
                   <p>SVM Model Prediction: {prediction['svm_model_prediction']}</p>"""

# Define inputs for Gradio with custom styling
inputs = [
    gr.Dropdown(choices=symptoms, label="Select Symptom 1"),
    gr.Dropdown(choices=symptoms, label="Select Symptom 2"),
    gr.Dropdown(choices=symptoms, label="Select Symptom 3")
]

# Launch Gradio interface
gr.Interface(fn=interface, inputs=inputs, outputs="html", title="Disease Predictor").launch()
