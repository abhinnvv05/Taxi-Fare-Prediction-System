import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('taxi_fare_model.pkl')

# Get user inputs
trip_duration = float(input("Enter trip duration (minutes): "))
distance_traveled = float(input("Enter distance traveled: "))
num_of_passengers = int(input("Enter number of passengers: "))
fare = float(input("Enter base fare: "))
tip = float(input("Enter tip amount: "))
miscellaneous_fees = float(input("Enter miscellaneous fees: "))
surge_applied = float(input("Enter surge multiplier: "))

# Create a DataFrame for prediction
new_data = pd.DataFrame({
    'trip_duration': [trip_duration],
    'distance_traveled': [distance_traveled],
    'num_of_passengers': [num_of_passengers],
    'fare': [fare],
    'tip': [tip],
    'miscellaneous_fees': [miscellaneous_fees],
    'surge_applied': [surge_applied]
})

# Predict total fare
predicted_fare = model.predict(new_data)
print(f"\nPredicted Total Fare: {predicted_fare[0]:.2f}")
