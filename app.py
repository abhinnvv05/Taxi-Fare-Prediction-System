import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('taxi_fare_model.pkl')

# Streamlit app title
st.title("Taxi Fare Prediction")

# Get user inputs
trip_duration = st.number_input("Enter Trip Duration (minutes):", min_value=0.0, step=0.1)
distance_traveled = st.number_input("Enter Distance Traveled (km):", min_value=0.0, step=0.1)
num_of_passengers = st.number_input("Enter Number of Passengers:", min_value=1, step=1)
fare = st.number_input("Enter Base Fare (₹):", min_value=0.0, step=0.1)
tip = st.number_input("Enter Tip Amount (₹):", min_value=0.0, step=0.1)
miscellaneous_fees = st.number_input("Enter Miscellaneous Fees (₹):", min_value=0.0, step=0.1)
surge_applied = st.number_input("Enter Surge Multiplier:", min_value=1.0, step=0.1)

# Predict button
if st.button("Predict Fare"):
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
    
    # Display the result
    st.success(f"Predicted Total Fare: ₹{predicted_fare[0]:.2f}")
