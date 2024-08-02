# model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset from CSV
df = pd.read_csv('data\data.csv')

# Define features and target variable
X = df[['feature1', 'feature2']]
y = df['target']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'model.joblib')

print("Model trained and saved as 'model.joblib'.")
