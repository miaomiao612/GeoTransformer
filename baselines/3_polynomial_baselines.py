import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA

# Load GDP data
with open('GDP_results.json', 'r') as file:
    gdp_data = json.load(file)
gdp_list = [{'tile_name': k, 'GDP': v['F2019GDP']} for k, v in gdp_data.items()]
gdp_df = pd.DataFrame(gdp_list)

# with open('Trips_results.json', 'r') as file:
#     trip_data = json.load(file)
# trip_list = [{'tile_name': k, 'GDP': v['Trips']} for k, v in trip_data.items()]
# gdp_df = pd.DataFrame(trip_list)

# Load latents
latents = pd.read_csv('latent_space.csv', index_col=0)
# Join datasets
data = latents.join(gdp_df.set_index('tile_name'))
X = data.drop(columns=['GDP']).values
y = data['GDP'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#---------------------------------------Polynomial Features----------------------------------#
# Apply PCA for dimensionality reduction
pca = PCA(n_components=128)  # Adjust n_components as needed
X_reduced = pca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Create polynomial features
poly_degree = 2
poly = PolynomialFeatures(degree=poly_degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train Lasso regression on polynomial features
alpha = 0.5
lasso_poly = Lasso(alpha=alpha)
lasso_poly.fit(X_train_poly, y_train)
y_train_pred_poly = lasso_poly.predict(X_train_poly)
y_test_pred_poly = lasso_poly.predict(X_test_poly)

# Calculate R², MSE, and MAE
train_r2_poly = r2_score(y_train, y_train_pred_poly)
test_r2_poly = r2_score(y_test, y_test_pred_poly)
train_mse_poly = mean_squared_error(y_train, y_train_pred_poly)
test_mse_poly = mean_squared_error(y_test, y_test_pred_poly)
train_mae_poly = mean_absolute_error(y_train, y_train_pred_poly)
test_mae_poly = mean_absolute_error(y_test, y_test_pred_poly)

# Print results
print(f"\nPolynomial Regression (degree={poly_degree}) - Alpha: {alpha}")
print(f"Train R²: {train_r2_poly}")
print(f"Test R²: {test_r2_poly}")
print(f"Train MSE: {train_mse_poly}")
print(f"Test MSE: {test_mse_poly}")
print(f"Train MAE: {train_mae_poly}")