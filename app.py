import joblib
import pandas as pd
import streamlit as st

# Load the model and feature names
model = joblib.load('car_mod.pkl')
features = joblib.load('car_col.pkl')

# Function to make predictions
def predict(new_data):
    pred = model.predict(new_data)
    return pred

# Streamlit UI components
st.title("Car Milage Model")

# Input fields for user to enter data
car_year = st.number_input("Enter year:")
car_kilometer= st.number_input("Enter kilometer driven:")
car_engine = st.number_input("Enter engine:")
car_power= st.number_input("Enter power:")
car_seats= st.number_input("Enter seats:")
car_fuel= st.text_input("Enter fuel type:")

car_transmission= st.text_input("Enter transmission:")
car_brands= st.text_input("Enter brand:")

# Predict button
if st.button("Predict"):
    # Create a dictionary from input data
    input_data = {
        'Year': [car_year],  # Wrap in list to match DataFrame structure
        'Kilometers_Driven': [car_kilometer],
        'Engine': [car_engine],
        'Power': [car_power],
        'Seats': [car_seats],
        'Fuel_Type': [car_fuel],
        'Transmission': [car_transmission],
        'brand': [car_brands]
    }
    
    # Convert dictionary to DataFrame
    new_data = pd.DataFrame(input_data)
    new_data=pd.get_dummies(data=new_data)


    for i in features:
        if i not in new_data.columns:
            new_data[i]=0
    
#     # Make prediction
    prediction = predict(new_data[features])  # Assuming 'features' contains the column names in the right order
    
    st.write(new_data[features])# to print new_data table according to column order 
    
#     # Display prediction result
    st.write(f"The predicted mileage is: {prediction[0]}")