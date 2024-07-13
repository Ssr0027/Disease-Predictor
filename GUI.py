import tkinter as tk
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Function to predict disease based on symptoms input by the user
def predictDisease(symptoms):
    # Update the path to your new data file
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

def predict_disease():
    # Retrieve symptoms from entry fields
    symptom1 = entry_symptom1.get()
    symptom2 = entry_symptom2.get()
    symptom3 = entry_symptom3.get()

    # Concatenate symptoms separated by commas
    user_input = f"{symptom1},{symptom2},{symptom3}"

    # Predict disease based on user input
    predictions = predictDisease(user_input)

    # Display predicted disease and message
    label_result.config(text=f"Predicted Disease: {predictions['final_prediction']}\nPlease see a doctor.")

# Create main window
root = tk.Tk()
root.title("Disease Predictor")

# Create labels for symptoms
label_symptom1 = tk.Label(root, text="Symptom 1:")
label_symptom1.grid(row=0, column=0, padx=10, pady=5)
label_symptom2 = tk.Label(root, text="Symptom 2:")
label_symptom2.grid(row=1, column=0, padx=10, pady=5)
label_symptom3 = tk.Label(root, text="Symptom 3:")
label_symptom3.grid(row=2, column=0, padx=10, pady=5)

# Create entry fields for symptoms
entry_symptom1 = tk.Entry(root)
entry_symptom1.grid(row=0, column=1, padx=10, pady=5)
entry_symptom2 = tk.Entry(root)
entry_symptom2.grid(row=1, column=1, padx=10, pady=5)
entry_symptom3 = tk.Entry(root)
entry_symptom3.grid(row=2, column=1, padx=10, pady=5)

# Create Predict button
button_predict = tk.Button(root, text="Predict", command=predict_disease)
button_predict.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Create label for displaying result
label_result = tk.Label(root, text="")
label_result.grid(row=4, column=0, columnspan=2, padx=10, pady=5)

# Run the GUI
root.mainloop()
