import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load the data
def load_data(file):
    return pd.read_excel(file)

# Function for preprocessing
def preprocess_data(cardf):
    cardf = cardf.drop(cardf.columns[0], axis=1)
    cardf['age'] = 2024 - cardf["Year"]
    cardf = cardf.drop(["Year"], axis=1)
    
    numcols = cardf[['Mileage', 'Engine', 'Power', 'Seats', 'age', 'Kilometers_Driven', 'Price']]
    objcols = cardf[['Location', 'Fuel_Type', 'Transmission', 'Owner_Type']]
    
    # Fill missing values in numerical columns
    for col in numcols.columns:
        numcols[col] = numcols[col].fillna(numcols[col].mean())
    
    objcols_dummy = pd.get_dummies(objcols, columns=['Location', 'Fuel_Type', 'Transmission', 'Owner_Type'])
    cardf_final = pd.concat([numcols, objcols_dummy], axis=1)
    
    return cardf_final, numcols, objcols_dummy

# Streamlit UI
st.title("Car Price Prediction")

# Upload training data
train_file = st.file_uploader("Upload Training Data (Excel)", type=["xlsx"])
if train_file is not None:
    train_df = load_data(train_file)
    st.write("Training Data", train_df.head())
    
    # Preprocess training data
    cardf_final_train, numcols_train, objcols_dummy_train = preprocess_data(train_df)
    
    # Prepare X and y
    X_train = cardf_final_train.drop(["Price"], axis=1)
    y_train = cardf_final_train['Price']
    
    # Train the model
    regmodel = LinearRegression().fit(X_train, np.log(y_train))
    
    # Display model score
    st.write(f"Model R^2 Score: {regmodel.score(X_train, np.log(y_train))}")
    
    # Upload test data
    test_file = st.file_uploader("Upload Test Data (Excel)", type=["xlsx"])
    if test_file is not None:
        test_df = load_data(test_file)
        st.write("Test Data", test_df.head())
        
        # Preprocess test data
        cardf_final_test, numcols_test, objcols_dummy_test = preprocess_data(test_df)
        
        # Prepare X_test for prediction
        X_test = cardf_final_test.drop(["Price"], axis=1)
        
        # Predict prices
        predicted_log_prices = regmodel.predict(X_test)
        predicted_prices = np.exp(predicted_log_prices)
        
        # Display predictions
        predictions_df = test_df[['Location', 'Fuel_Type', 'Transmission', 'Owner_Type']]
        predictions_df['Predicted Price'] = predicted_prices
        
        st.write("Predicted Prices", predictions_df)
        
        # Show RMSE
        rmse = np.sqrt(mean_squared_error(test_df['Price'], predicted_prices))
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")
