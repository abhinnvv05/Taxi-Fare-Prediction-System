import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dataset
print("Loading dataset...")
data = pd.read_csv(r'C:\\miniproject\\taxi_fare\\train.csv')

# Preprocessing
print("Preprocessing data...")

# Handle missing or invalid data
data = data.dropna()  # Drop rows with NaN values
data = data[data['total_fare'] > 0]  # Remove rows with non-positive total fares

# Select relevant features
features = ['trip_duration', 'distance_traveled', 'num_of_passengers', 'fare', 'tip', 'miscellaneous_fees', 'surge_applied']
target = 'total_fare'

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
print("Training the model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
print("Evaluating the model...")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save the model (optional)
import joblib
joblib.dump(model, 'taxi_fare_model.pkl')

print("Model training and evaluation complete.")
