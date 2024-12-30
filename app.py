
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import nbformat
from nbconvert import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/knn_model.pkl')

@app.route('/')
def home():
    return render_template('Cricket.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create a DataFrame with the input features
        input_data = pd.DataFrame({
            'Overs_Played': [float(data['overs'])],
            'Wickets_Lost': [int(data['wickets'])],
            'Run_Rate': [float(data['Run_Rate'])],
            'Home_Away': [data['Home_Away']],
            'Opponent_Strength': [data['Opponent_Strength']],
            'Pitch_Condition': [data['Pitch_Condition']],
            'Weather': [data['Weather']]
        })

        # Make prediction
        prediction = model.predict(input_data)
        
        # Round to 2 decimal places
        predicted_score = round(float(prediction[0]), 2)
        
        return jsonify({'predicted_Score': predicted_score})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)