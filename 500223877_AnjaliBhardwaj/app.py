# app.py
#importing necessary libraries

from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Loading the trained model
iot = joblib.load('500223877_AnjaliBhardwaj/data.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get  data
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    
    # Prepare the feature vector
    features = np.array([[feature1, feature2]])
    
    # Make prediction
    prediction = iot.predict(features)
    
    # Return prediction on the web page
    return render_template('index.html', prediction=prediction[0])

if __name__== '__main__':
    app.run(debug=True)
