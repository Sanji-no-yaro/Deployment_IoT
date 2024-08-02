# generate_data.py
import pandas as pd
import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Generate random data
num_samples = 100
feature1 = np.random.rand(num_samples) * 100
feature2 = np.random.rand(num_samples) * 200
# Assuming a simple linear relation with some noise
target = 3 * feature1 + 2 * feature2 + np.random.randn(num_samples) * 10

# Create a DataFrame
df = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'target': target
})

# Save to CSV
df.to_csv('data.csv', index=False)

print("Random data generated and saved as 'data.csv'.")