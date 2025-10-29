from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model_path = "E:/Suicidal_prediction/model/suicide_pipeline.joblib"
model = joblib.load(model_path)

# list of categorical fields (order must match train pipeline)
categorical_fields = ['Gender', 'Stress Level', 'Academic Performance',
                      'Health Condition', 'Relationship Condition', 'Family Problem',
                      'Depression Level', 'Anxiety Level', 'Mental Support', 'Self Harm Story']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read form fields
        age = float(request.form.get('Age', 23))
        # Collect categorical values in same order used for training
        input_dict = {
            'Age': [age]
        }
        for f in categorical_fields:
            input_dict[f] = [request.form.get(f, '')]

        X = pd.DataFrame(input_dict)
        proba = model.predict_proba(X)[:,1][0]
        percent = round(proba * 100, 2)
        risk = "Low" if proba < 0.33 else ("Moderate" if proba < 0.66 else "High")

        return render_template('result.html', percent=percent, risk=risk)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
