import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


iowa_file_path = "C:/Users/cntri/Downloads/train.csv"

home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

feature_names = feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

X = home_data[feature_names]

homeModel = DecisionTreeRegressor(random_state=1090)

homeModel.fit(X,y)

predictions = homeModel.predict(X)

print(mean_absolute_error(y, predictions))


