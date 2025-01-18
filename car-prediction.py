import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load the data
def load_data(file):
    return pd.read_excel(file)

# Function for preprocessing
def preprocess_data(cardf, train_cols=None):
    cardf = cardf.drop(cardf.columns[0], axis=1)
    cardf['age'] = 2024 - cardf["Year"]
    cardf = cardf.drop(["Year"], axis=1)
    
    numcols = cardf[['Mileage', 'Engine', 'Power', 'Seats', 'age', 'Kilometers_Driven', 'Price']]
    objcols = cardf[['Location', 'Fuel_Type', 'Transmission', 'Owner_Type']]
    
    # Fill missing values in numerical columns
    for col in numcols.columns:
        numcols[col] = numcols[col].fillna(numcols[col].mean())
    
    objcols_dummy = pd.get_dummies(objcols, columns=['Location', 'Fuel_Type', 'Transmission', 'Owner_Type'])
    
    # If this is the training data, store the columns to use in the test data
    if train_cols is None:
        train_cols = objcols_dummy.columns
    
    # Ensure the test data has the same columns as the training data
    objcols_dummy = objcols_dummy.reindex(columns=train_cols, fill_value=0)
    
    cardf_final = pd.concat([numcols, objcols_dummy], axis=1)
    
    return cardf_final, train_cols

# Streamlit UI
st.title("Car Price Prediction")

# Upload training data
train_file = st.file_uploader("Upload Training Data (Excel)", type=["xlsx"])
train_cols = None  # Variable to store columns for one-hot encoding
if train_file is not None:
    train_df = load_data(train_file)
    st.write("Training Data", train_df.head())
    
    # Preprocess training data
    cardf_final_train, train_cols = preprocess_data(train_df)
    
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
        cardf_final_test, _ = preprocess_data(test_df, train_cols)
        
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
