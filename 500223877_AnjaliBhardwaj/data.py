# model.py, 
#importing necessary library
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset from CSV
df = pd.read_csv('/Users/anjalibhardwaj/Desktop/deployment of ai solution/Deployment_IoT/500223877_AnjaliBhardwaj/sample_data/data.csv')

# Define features and target variable
X = df[['feature1', 'feature2']]
y = df['target']

# Training the model
model = LinearRegression()
model.fit(X, y)

# Saving  the trained model
joblib.dump(model, 'data.joblib')

print("saved the trained model as  'data.joblib'.")

