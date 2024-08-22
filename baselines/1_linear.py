#########################################################################
##   This file is part of the GeoTransformer                           ##
##                                                                     ##
##   Copyright (C) 2024 The GeoTransformer Team                        ##
##   Primary contacts: Yuhao Jia <yuhaojia98@ucla.edu>                 ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Load the data
acs_file = 'Acs_results.json'
gdp_file = 'Trips_results.json'

def load_data(acs_file, gdp_file):
    with open(acs_file, 'r') as f:
        acs_data = json.load(f)
    with open(gdp_file, 'r') as f:
        gdp_data = json.load(f)
    
    X = []
    y = []
    
    for key in gdp_data:
        if key in acs_data:
            features = list(acs_data[key].values())
            target = gdp_data[key]['Trips']
            X.append(features)
            y.append(target)
    
    # Convert X and y to numpy arrays
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
    # Remove rows with NaN values
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    
    return X, y

X, y = load_data(acs_file, gdp_file)

# Split the data into training and testing sets
test_size = 0.2  # Example test size; you can adjust this
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate MSE, MAE, and R2 for the training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate MSE, MAE, and R2 for the testing set
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print the results
print(f'Training MSE: {train_mse:.4f}')
print(f'Training MAE: {train_mae:.4f}')
print(f'Training R2: {train_r2:.4f}')
print(f'Testing MSE: {test_mse:.4f}')
print(f'Testing MAE: {test_mae:.4f}')
print(f'Testing R2: {test_r2:.4f}')
