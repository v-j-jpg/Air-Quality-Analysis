import HealthDataAnalys
import AirQualityAnalysis
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import math

##################################
merged_data = pd.merge(AirQualityAnalysis.combined_air_quality_data,HealthDataAnalys.combined_health_df, on=['GeoID', 'IndicatorID', 'Year', 'Geography'], how='inner')

merged_df = merged_data.groupby(['GeoID', 'GeoRank', 'Geography', 'Indicator','Mean','IndicatorID', 'Year']).agg({'MeanDeaths':'first', 'MeanAsthma':'first'}).reset_index()


X_features = merged_df[['Year', 'GeoID', 'IndicatorID']]
y_targets = merged_df[['Mean','MeanDeaths','MeanAsthma']]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_targets, test_size=0.2, random_state=42)
X_train_filled = X_train.fillna(0)
y_train_filled = y_train.fillna(0)

base_regressor = RandomForestRegressor()

multioutput_regressor = MultiOutputRegressor(base_regressor)

multioutput_regressor.fit(X_train_filled, y_train_filled)

y_pred = multioutput_regressor.predict(X_test)
predictions_df = pd.DataFrame(y_pred, columns=['predicted_mean','predicted_deaths', 'predicted_asthma_visits'])
predictions_df['IndicatorID'] = X_test['IndicatorID'].values
predictions_df['predicted_mean'] = predictions_df['predicted_mean'].apply(lambda x: math.ceil(x))
predictions_df['predicted_deaths'] = predictions_df['predicted_deaths'].apply(lambda x: math.ceil(x))
predictions_df['predicted_asthma_visits'] = predictions_df['predicted_asthma_visits'].apply(lambda x: math.ceil(x))

print(predictions_df)

