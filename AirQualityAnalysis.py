import pandas as pd
import itertools
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import numpy as np
import itertools
import seaborn as sns
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform


ozone_data = pd.read_csv(r'C:\Users\Jo\Desktop\Big Data\Ozone (O3).csv')
pm_data = pd.read_csv(r'C:\Users\Jo\Desktop\Big Data\Fine Particles (PM 2.5).csv')
nitrogen_data = pd.read_csv(r'C:\Users\Jo\Desktop\Big Data\Nitrogen dioxide (NO2).csv')

combined_air_quality_data = pd.concat([ozone_data, pm_data, nitrogen_data])

combined_air_quality_data['Year'] = pd.to_datetime(combined_air_quality_data['TimePeriod'].str.extract('(\d{4})')[0]).dt.year
combined_air_quality_data = combined_air_quality_data.sort_values('Year')
combined_air_quality_data.to_csv(r'C:\Users\Jo\Desktop\Big Data\combined_air_quality_data.csv', index=False)


X = combined_air_quality_data[['GeoID', 'IndicatorID', 'Year']]  # Varijable koje utiču na predikciju
y = combined_air_quality_data['Mean']  # Ciljne varijable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'learning_rate': [0.01, 0.05, 0.1]
}

model = GradientBoostingRegressor(random_state=42)
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=5, random_state=42)
random_search.fit(X_train, y_train)

y_pred = random_search.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")


#######################################################

predictions_df_2023 = pd.DataFrame(list(itertools.product(combined_air_quality_data['GeoID'].unique(), combined_air_quality_data['IndicatorID'].unique())), columns=['GeoID', 'IndicatorID'])
predictions_df_2023['Year'] = 2023

predictions_2023 = random_search.predict(predictions_df_2023[['GeoID', 'IndicatorID', 'Year']])

predictions_df_2023['PredictedMean'] = predictions_2023

predictions_df_2023.to_csv(r'C:\Users\Jo\Desktop\Big Data\predictions_df_2023.csv', index=False)
max_indices = predictions_df_2023.groupby('IndicatorID')['PredictedMean'].idxmax()
min_indices = predictions_df_2023.groupby('IndicatorID')['PredictedMean'].idxmin()

most_polluted_geo_2023 = predictions_df_2023.loc[max_indices]
least_polluted_geo_2023 = predictions_df_2023.loc[min_indices]

print("Most polluted location in 2023:")
print(most_polluted_geo_2023[['IndicatorID', 'GeoID', 'PredictedMean']])

print("Least polluted locations in 2023:")
print(least_polluted_geo_2023[['IndicatorID', 'GeoID', 'PredictedMean']])

for year in range(2023, 2027):
    predictions_df_yearly = predictions_df_2023.copy()

    predictions_df_yearly['Year'] = year

    predictions_yearly = random_search.predict(predictions_df_yearly[['GeoID', 'IndicatorID', 'Year']])

    predictions_df_yearly['PredictedMean'] = predictions_yearly

    predictions_df_2023 = pd.concat([predictions_df_2023, predictions_df_yearly], ignore_index=True)

grouped_predictions = predictions_df_2023.groupby(['Year', 'IndicatorID'])['PredictedMean'].mean().reset_index()
predictions_df_2023.to_csv(r'C:\Users\Jo\Desktop\Big Data\predictions_df_2023.csv', index=False)

###########################################################################
#predictions_df_2023 = predictions_df_2023.rename(columns={'PredictedMean': 'Mean'})

#combined_air_quality_data_with_predictions = pd.concat([combined_air_quality_data, predictions_df_2023[['IndicatorID', 'GeoID', 'Year', 'Mean']]], ignore_index=True)

indicator_names = {385: 'O3', 375: 'NO2', 365: 'PM2.5'}

combined_air_quality_data = combined_air_quality_data.groupby(['IndicatorID', 'Year'])['Mean'].mean().reset_index()

pivot_data = combined_air_quality_data.pivot(index='Year', columns='IndicatorID', values='Mean')

diff_data = pivot_data.sub(pivot_data.loc[2009], axis=1).mul(100).div(pivot_data.loc[2009])

plt.figure(figsize=(12, 8))
for indicator_id in diff_data.columns:
    plt.plot(diff_data.index, diff_data[indicator_id], label=f"IndicatorID {indicator_id}", linestyle='--')


plt.xlabel('Year')
plt.ylabel('Percentage Change %')
plt.title('Percentage Change of Mean Deaths and Mean Asthma for Each IndicatorID')
plt.legend(title='IndicatorID', loc='right')
plt.grid(True)
plt.tight_layout()
plt.show()
