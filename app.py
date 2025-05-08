import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template

app = Flask(__name__)

# Load and train the model once on startup
def load_model():
    try:
        df = pd.read_csv("breast-cancer-data.csv")
    except FileNotFoundError:
        raise SystemExit("Error: 'breast-cancer-data.csv' not found in the project directory.")

    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean']
    x = df[features]
    y = df['diagnosis']

    model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(x, y)
    return model, features

model, features = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[feature]) for feature in features]
        input_df = pd.DataFrame([input_data], columns=features)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            result = 'The patient is diagnosed with breast cancer.'
            confidence = f'Confidence: {probability * 100:.2f}%'
        else:
            result = 'The patient is not diagnosed with breast cancer.'
            confidence = f'Confidence: {(1 - probability) * 100:.2f}%'

        # Show result on a separate page
        return render_template('result.html', prediction=result, confidence=confidence)
    
    except Exception as e:
        return render_template('index.html', prediction=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
