# Importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_PATH = "/Users/shubham/Desktop/DiseasePredictor/Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Training the models
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

symptoms = X.columns.values

data_dict = {
    "symptom_index": {symptom: index for index, symptom in enumerate(symptoms)},
    "predictions_classes": encoder.classes_
}

# Function to predict disease based on symptoms input by the user
def predictDisease(symptoms):
    # Converting user input symptoms to lowercase and splitting them
    symptoms = symptoms.lower().split(",")
    
    # Creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"].get(symptom)
        if index is not None:
            input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    # Generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    
    # Making final prediction by taking mode of all predictions
    final_prediction = max(set([rf_prediction, nb_prediction, svm_prediction]), key=[rf_prediction, nb_prediction, svm_prediction].count)
    
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions

# Getting symptom inputs from the user
user_input = input("Enter the symptoms (comma-separated): ")

# Predicting disease based on user input
predictions = predictDisease(user_input)
print("Predicted Disease:", predictions["final_prediction"])
