import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import itertools
import seaborn as sns
import math


deaths_pm25 = pd.read_csv("Deaths due to PM2.5.csv")
deaths_ozone = pd.read_csv("Cardiac and respiratory deaths due to Ozone.csv")
asthma_visits_pm25 = pd.read_csv("Asthma emergency department visits due to PM2.5.csv")
asthma_visits_ozone = pd.read_csv("Asthma emergency departments visits due to Ozone.csv")

combined_health_data = pd.concat([deaths_pm25, deaths_ozone, asthma_visits_pm25,asthma_visits_ozone])

def extract_years(time_period):
    years = time_period.split('-')
    return int(years[0]), int(years[1])

combined_health_data['StartYear'], combined_health_data['EndYear'] = zip(*combined_health_data['TimePeriod'].map(extract_years))

dfs = []
asthma_visits_df = []
combined_health_df =[]
combined_health_data['Estimated annual number'] = combined_health_data['Estimated annual number'].apply(lambda x: float(x.replace(',', '')) if isinstance(x, str) and ',' in x else x)

for index, row in combined_health_data.iterrows():
    num_years = row['EndYear'] - row['StartYear'] + 1
    number = row['Estimated annual number']
    yearly_mean = float(number) / num_years
    if yearly_mean !=0 and pd.notnull(number):
        if row['ResultID'] == 1:
            yearly_data = pd.DataFrame({'Year': range(row['StartYear'], row['EndYear'] + 1), 'MeanDeaths': yearly_mean, 'IndicatorID': row['IndicatorID'], 'GeoID' : row['GeoID'], 'Geography': row['Geography']})
            dfs.append(yearly_data)
            combined_health_df.append(yearly_data)
        if row['ResultID'] == 2:
            yearly_data = pd.DataFrame({'Year': range(row['StartYear'], row['EndYear'] + 1), 'MeanAsthma': yearly_mean, 'IndicatorID': row['IndicatorID'], 'GeoID': row['GeoID'],'Geography': row['Geography']})
            asthma_visits_df.append(yearly_data)
            combined_health_df.append(yearly_data)

combined_health_df = pd.concat(combined_health_df, ignore_index=True)
mean_deaths = pd.concat(dfs, ignore_index=True)
mean_asthma = pd.concat(asthma_visits_df, ignore_index=True)
mean_deaths = mean_deaths.dropna(subset=['GeoID'])
mean_asthma = mean_asthma.dropna(subset=['GeoID'])
mean_deaths['GeoID'] = mean_deaths['GeoID'].astype(int)
mean_asthma['GeoID'] = mean_asthma['GeoID'].astype(int)
indicator_names = {385: 'O3', 375: 'NO2', 365: 'PM2.5'}

mean_deaths_per_year = mean_deaths.groupby('Year')['MeanDeaths'].mean()
mean_asthma_per_year = mean_asthma.groupby('Year')['MeanAsthma'].mean()

plt.figure(figsize=(12, 6))
plt.plot(mean_deaths_per_year.index, mean_deaths_per_year.values, label='Mean Deaths', color='blue')
plt.plot(mean_asthma_per_year.index, mean_asthma_per_year.values, label='Mean Asthma Visits', color='green')

handles = [plt.Line2D([], [], color='blue', label='Mean Deaths'),
           plt.Line2D([], [], color='green', label='Mean Asthma Visits')]
plt.legend(handles=handles)

plt.xlabel('Year')
plt.ylabel('Average number')
plt.title('Average number of deaths and emergency visits because of asthma')
plt.grid(True)
plt.show()

mean_deaths.to_csv('mean_deaths.csv', index=False)

X_deaths = mean_deaths[['Year', 'GeoID', 'IndicatorID']]
y_deaths = mean_deaths['MeanDeaths']
y_deaths = pd.to_numeric(y_deaths, errors='coerce')
X_deaths_train, X_deaths_test, y_deaths_train, y_deaths_test = train_test_split(X_deaths, y_deaths, test_size=0.2, random_state=42)

X_asthma = mean_asthma[['Year', 'GeoID', 'IndicatorID']]
y_asthma = mean_asthma['MeanAsthma']
X_asthma_train, X_asthma_test, y_asthma_train, y_asthma_test = train_test_split(X_asthma, y_asthma, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Pretrage hiperparametara
grid_search.fit(X_deaths_train, y_deaths_train)


best_rf1 = grid_search.best_estimator_

y_deaths_pred = best_rf1.predict(X_deaths_test)

mse = mean_squared_error(y_deaths_test, y_deaths_pred)
r2 = r2_score(y_deaths_test, y_deaths_pred)

print(f"Mean Squared Error deaths: {mse}")
print(f"R-squared deaths: {r2}")


grid_search.fit(X_asthma_train, y_asthma_train)

best_rf = grid_search.best_estimator_

y_asthma_pred = best_rf.predict(X_asthma_test)

# Evaluacija greske
mse = mean_squared_error(y_asthma_test, y_asthma_pred)
r2 = r2_score(y_asthma_test, y_asthma_pred)
print(f"Mean Squared Error Asthma: {mse}")
print(f"R-squared Asthma: {r2}")

################################################
unique_combinations = mean_deaths[['Year', 'GeoID', 'IndicatorID']]
unique_combinations.loc[:, 'Year'] = 2020

predicted_values = best_rf1.predict(unique_combinations)
unique_combinations['PredictedDeaths'] = predicted_values

X_2020 = mean_deaths[['Year', 'GeoID', 'IndicatorID']]
X_2020.loc[:, 'Year'] = 2020

y_2020_pred = best_rf1.predict(X_2020)

predictions_death_2020_df = pd.DataFrame(list(itertools.product(mean_asthma['GeoID'].unique(), mean_asthma['IndicatorID'].unique())), columns=['GeoID', 'IndicatorID'])
predictions_death_2020_df['Year'] = 2020
predictions_death_2020_df['PredictedDeaths'] = best_rf1.predict(predictions_death_2020_df[['Year', 'GeoID', 'IndicatorID']])

predictions_death_2020_df = predictions_death_2020_df.dropna(subset=['GeoID'])
predictions_death_2020_df['GeoID'] = predictions_death_2020_df['GeoID'].astype(int)
predictions_death_2020_df['PredictedDeaths'] = predictions_death_2020_df['PredictedDeaths'].apply(lambda x: math.ceil(x))
predictions_death_2020_df = pd.merge(predictions_death_2020_df, mean_deaths[['GeoID', 'Geography']], on='GeoID', how='left')

most_polluted_2020 = predictions_death_2020_df.loc[predictions_death_2020_df.groupby('IndicatorID')['PredictedDeaths'].idxmax()]
least_polluted_2020 = predictions_death_2020_df.loc[predictions_death_2020_df.groupby('IndicatorID')['PredictedDeaths'].idxmin()]
grouped_data = predictions_death_2020_df.groupby(['GeoID', 'IndicatorID']).first().reset_index()
print(grouped_data)
grouped_data.to_csv('predicted_deaths.csv')
print("Most deaths by indicator in  2020: \n", most_polluted_2020)
print("Least deaths by indicator in  2020: \n", least_polluted_2020)
#################################################################

predictions_asthma_2020 = pd.DataFrame(list(itertools.product(mean_asthma['GeoID'].unique(), mean_asthma['IndicatorID'].unique())), columns=['GeoID', 'IndicatorID'])
predictions_asthma_2020['Year'] = 2020
predictions_asthma_2020 = predictions_asthma_2020.dropna(subset=['GeoID'])

predictions_asthma_2020['GeoID'] = predictions_asthma_2020['GeoID'].astype(int)
predictions_asthma_2020['IndicatorID'] = predictions_asthma_2020['IndicatorID'].astype(int)
predictions_asthma_2020['Year'] = predictions_asthma_2020['Year'].astype(int)
predictions_asthma_2020 = pd.merge(predictions_asthma_2020, mean_asthma[['GeoID', 'Geography']], on='GeoID', how='left')

predictions_asthma_2020['PredictedAsthmaVisits'] = best_rf.predict(predictions_asthma_2020[['Year', 'GeoID', 'IndicatorID']])
predictions_asthma_2020['PredictedAsthmaVisits'] = predictions_asthma_2020['PredictedAsthmaVisits'].apply(lambda x: math.ceil(x))
grouped_data = predictions_asthma_2020.groupby(['GeoID', 'IndicatorID']).first().reset_index()
print(grouped_data)
grouped_data.to_csv('predicted_asthma.csv')

max_geoID = predictions_asthma_2020.loc[predictions_asthma_2020.groupby('IndicatorID')['PredictedAsthmaVisits'].idxmax()]
min_geoID = predictions_asthma_2020.loc[predictions_asthma_2020.groupby('IndicatorID')['PredictedAsthmaVisits'].idxmin()]

print("Most asthma visits by indicator in 2020: \n" , max_geoID)
print("\nLeast asthma visits by indicator in 2020: \n", min_geoID)

#######################################

mean_deaths = mean_deaths.groupby(['Year', 'IndicatorID'])['MeanDeaths'].mean().reset_index()


plt.figure(figsize=(12, 8))
sns.lineplot(x='Year', y='MeanDeaths', hue='IndicatorID', data=mean_deaths)
plt.title('Trends in mortality by year for different indicators')
plt.xlabel('Year')
plt.ylabel('Mean annual mortality')
plt.legend(title='IndicatorID')
plt.show()

############################################
predictions_death_2020_df = predictions_death_2020_df.rename(columns={'PredictedDeaths': 'MeanDeaths'})
#mean_deaths = pd.concat([mean_deaths, predictions_death_2020_df[['Year', 'IndicatorID', 'MeanDeaths']]], ignore_index=True)

predictions_asthma_2020 = predictions_asthma_2020.rename(columns={'PredictedAsthmaVisits': 'MeanAsthma'})
#mean_asthma = pd.concat([mean_asthma, predictions_asthma_2020[['Year', 'IndicatorID', 'MeanAsthma', 'GeoID']]], ignore_index=True)

mean_deaths_yearly = mean_deaths.groupby(['IndicatorID', 'Year'])['MeanDeaths'].mean().reset_index()

pivot_data_deaths = mean_deaths_yearly.pivot(index='Year', columns='IndicatorID', values='MeanDeaths')

diff_data_deaths = pivot_data_deaths.sub(pivot_data_deaths.loc[2009], axis=1).mul(100).div(pivot_data_deaths.loc[2009])

mean_asthma_yearly = mean_asthma.groupby(['Year','IndicatorID', ])['MeanAsthma'].mean().reset_index()

pivot_data_asthma = mean_asthma_yearly.pivot(index='Year', columns='IndicatorID', values='MeanAsthma')

diff_data_asthma = pivot_data_asthma.sub(pivot_data_asthma.loc[2009], axis=1).mul(100).div(pivot_data_asthma.loc[2009])

plt.figure(figsize=(12, 8))
for indicator_id in diff_data_deaths.columns:
    plt.plot(diff_data_deaths.index, diff_data_deaths[indicator_id], label=f"Mean Deaths - IndicatorID {indicator_id}", linestyle='--')
    if indicator_id in diff_data_asthma.columns:
        plt.plot(diff_data_asthma.index, diff_data_asthma[indicator_id], label=f"Mean Asthma - IndicatorID {indicator_id}")


plt.xlabel('Year')
plt.ylabel('Percentage Change (%)')
plt.title('Percentage Change of Mean Deaths and Mean Asthma for Each IndicatorID')
plt.legend(title='IndicatorID', loc='right')
plt.grid(True)
plt.tight_layout()
plt.show()

