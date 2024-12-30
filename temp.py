from flask import Flask, render_template, request, jsonify
import nbformat
from nbconvert import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Cricket.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    Overs_Played = data.get('overs')
    Wickets_Lost = data.get('wickets')
    Run_Rate = data.get('Run_Rate')
    Home_Away = data.get('Home_Away')
    Opponent_Strength = data.get('Opponent_Strength')
    Pitch_Condition = data.get('Pitch_Condition')
    Weather = data.get('Weather')

    # Call Jupyter Notebook to process the prediction logic
    prediction = execute_notebook(
        'model/Cricket.ipynb',
        {
            'overs': Overs_Played,
            'wickets': Wickets_Lost,
            'Run_Rate': Run_Rate,
            'Home_Away': Home_Away,
            'Opponent_Strength': Opponent_Strength,
            'Pitch_Condition': Pitch_Condition,
            'Weather': Weather
        }
    )
    return jsonify({'predicted_Score': prediction})

def execute_notebook(notebook_path, params):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Inject parameters into the notebook
    nb['cells'].insert(0, nbformat.v4.new_code_cell(f"params = {params}"))

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb)

    # Retrieve the output
    for cell in nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if 'text' in output:
                    return output['text'].strip()

if __name__ == '__main__':
    app.run(debug=True)
    