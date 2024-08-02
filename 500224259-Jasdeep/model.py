# model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('C:/Users/jotsa/OneDrive/Desktop/IoT/Deployment_IoT/500224259-Jasdeep/database/data1.csv')

X = df[['feature1', 'feature2']]
y = df['target']

# Training the model
model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'model.joblib')

print("Model trained and saved as 'model.joblib'.")