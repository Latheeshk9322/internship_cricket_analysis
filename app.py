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


# def predict():
#     try:
#         data = request.json
#         print("Received data:", data)  # Debugging line

#         # ... rest of the code remains the same
#     except Exception as e:
#         print("Error during prediction:", str(e))  # Debugging line
#         return jsonify({'error': str(e)}), 500


#         # Create a DataFrame with the input features
#         input_data = pd.DataFrame({
#             'Overs_Played': [float(data['overs'])],
#             'Wickets_Lost': [int(data['wickets'])],
#             'Run_Rate': [float(data['Run_Rate'])],
#             'Home_Away': [data['Home_Away']],
#             'Opponent_Strength': [data['Opponent_Strength']],
#             'Pitch_Condition': [data['Pitch_Condition']],
#             'Weather': [data['Weather']]
#         })

#         # Log the input data for debugging
#         print("Input data:", input_data)

#         # Make prediction
#         prediction = model.predict(input_data)

#         # Ensure the prediction is valid
#         if prediction is None or len(prediction) == 0:
#             return jsonify({'error': "Model returned no prediction."}), 500

#         # Round to 2 decimal places
#         predicted_score = round(float(prediction[0]), 2)

#         return jsonify({'predicted_Score': predicted_score})

#     except Exception as e:
#         # Log the error for debugging
#         print("Error during prediction:", str(e))
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)




def predict():
    try:
        # Parse incoming JSON request data
        data = request.json
        print("Received data:", data)  # Debugging output
        
        # Validate input data and convert to the correct format
        input_data = pd.DataFrame({
            'Overs_Played': [float(data['overs'])],
            'Wickets_Lost': [int(data['wickets'])],
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


