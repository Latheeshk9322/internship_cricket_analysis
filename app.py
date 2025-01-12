from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import nbformat
from nbconvert import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('model/knn_model.pkl')
except FileNotFoundError:
    raise FileNotFoundError("The model file 'knn_model.pkl' was not found. Ensure it is in the correct directory.")

@app.route('/')
def home():
    return render_template('Cricket.html')

@app.route('/predict', methods=['POST'])

def predict():
    try:
        # Parse incoming JSON request data
        data = request.json
        print("Received data:", data)  # Debugging output
        
        # Validate input data and convert to the correct format
        input_data = pd.DataFrame({
            'overs': [float(data['overs'])],
            'wickets': [int(data['wickets'])],
            'Run_Rate': [float(data['Run_Rate'])],
            'Home_Away': [int(data['Home_Away'])],
            'Opponent_Strength': [int(data['Opponent_Strength'])],
            'Pitch_Condition': [int(data['Pitch_Condition'])],
            'Weather': [int(data['Weather'])]
        })

        # Load the model and make a prediction
        prediction = model.predict(input_data)
        predicted_score = round(float(prediction[0]), 2)
        
        # Return the predicted score as a JSON response
        return jsonify({'predicted_Score': predicted_score})

    except Exception as e:
        # Log the exception and return an error response
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


