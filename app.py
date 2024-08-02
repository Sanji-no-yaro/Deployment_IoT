# app.py
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Machine Learning App"

@app.route('/upload', methods=['POST'])
def upload_data():
    file = request.files['file']
    data = pd.read_csv(file)
    return jsonify(data.head().to_dict())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    model = LinearRegression()
    X = df[['feature1', 'feature2']]
    y = df['target']
    model.fit(X, y)
    predictions = model.predict(X)
    return jsonify(predictions.tolist())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
