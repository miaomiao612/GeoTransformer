import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import Lasso


# create example encoder_output
encoder_output = np.random.rand(5, 4096)

names = ["tile_100_100.tif", "tile_100_101.tif", "tile_100_102.tif", "tile_100_103.tif", "tile_100_104.tif"]

encoder_output = pd.DataFrame(encoder_output, index=names)


# load gdp example data
with open('/content/drive/MyDrive/yuhao_jia/GDP_results.json', 'r') as file:
    gdp_data = json.load(file)



# make sure that gdp and code embeddings are consistent,,,
gdp_values = [gdp_data[tile_name]['F2019GDP'] for tile_name in encoder_output.index if tile_name in gdp_data]
y = np.array(gdp_values)

X_train, X_test, y_train, y_test = train_test_split(encoder_output, y, test_size=0.2, random_state=42)

results = []

alphas = (1e-3) * np.array([5, 6, 7, 8, 9, 10])
for a in alphas:
    lasso = Lasso(alpha=a)
    cv_results = cross_validate(lasso, X_train, y_train, cv=3, scoring='r2', 
                                return_train_score=True, return_estimator=True)
    results.append(cv_results)
    print(f"Alpha: {a}")
    print("Test Scores:", cv_results['test_score'])
    print("Train Scores:", cv_results['train_score'])
