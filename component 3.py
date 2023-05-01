#!/usr/bin/env python
# coding: utf-8

# # 1. Problem definition
# Testing the following hypotheses:
# 
# Hypothesis 1: The COVID-19 pandemic has significantly impacted healthcare infrastructure in both developed and developing countries.
# 
# Hypothesis 2: The COVID-19 pandemic has affected the availability of medical equipment and personnel in both developed and developing countries.
# 
# Hypothesis 3: Healthcare systems in both developed and developing countries have responded to the pandemic, but their responses have been limited by the severity of the crisis.

# # 2. Data description 
# The dataset used to solve these research questions is owid-covid-data.csv and for this research select only required features  which is related to hospital infrastructure.
# The below code imports pandas for data manupulation, matplotlib and seaborn visualization and reads the header of owid-covid-data. import warnings for to ignore any warnings while executing the program 

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Load data owid-covid-data
df = pd.read_csv('covid-data.csv')
df.head()


# Below code gives shape of our datasets to understand how many observations and features are there in our dataset

# In[13]:


print(df.shape)


# # 3. Feature engineering and data processing
# Date, location, total cases, new cases, total deaths, new deaths, hospital beds per thousand, total vaccines per hundred, total vaccinations per hundred, and human development index are the features used to answer these study questions. Preprocess the data by choosing pertinent columns, substituting the mean for missing values, and averaging the results by year:
# 

# In[14]:


# Convert date to datetime format
df['date'] = pd.to_datetime(df['date'])

# Select relevant columns
cols = ['date', 'location', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 
        'hospital_beds_per_thousand', 'total_vaccinations_per_hundred','people_vaccinated_per_hundred','human_development_index',
       'new_deaths_per_million','icu_patients_per_million']

# Create a new DataFrame with selected columns
df_selected = df[cols]

# Fill missing values with appropriate methods
df_selected['hospital_beds_per_thousand'] = df_selected['hospital_beds_per_thousand'].fillna(df_selected.groupby('location')['hospital_beds_per_thousand'].transform('mean'))
df_selected['total_vaccinations_per_hundred'] = df_selected['total_vaccinations_per_hundred'].fillna(df_selected.groupby('location')['total_vaccinations_per_hundred'].transform('mean'))
df_selected['people_vaccinated_per_hundred'] = df_selected['people_vaccinated_per_hundred'].fillna(df_selected.groupby('location')['people_vaccinated_per_hundred'].transform('mean'))
df_selected['human_development_index'] = df_selected['human_development_index'].fillna(df_selected.groupby('location')['human_development_index'].transform('mean'))
df_selected['new_deaths_per_million'] = df_selected['new_deaths_per_million'].fillna(df_selected.groupby('location')['new_deaths_per_million'].transform('mean'))
df_selected['icu_patients_per_million'] = df_selected['icu_patients_per_million'].fillna(df_selected.groupby('location')['icu_patients_per_million'].transform('mean'))

# Aggregate data by year
df_selected['year'] = df_selected['date'].dt.year
df_yearly = df_selected.groupby(['year', 'location']).agg({
    'total_cases': 'max',
    'total_deaths': 'max',
    'hospital_beds_per_thousand': 'mean',
    'total_vaccinations_per_hundred': 'mean',
    'people_vaccinated_per_hundred': 'mean',
    'human_development_index': 'mean',
    'icu_patients_per_million': 'mean',
    'new_deaths_per_million': 'mean'
}).reset_index()


# # 4. Results 
# Qusetion 1: Is COVID-19 pandemic has significantly impacted healthcare infrastructure in both developed and developing countries.
# 
# To make the visualization more informative, we can split the data into two groups: developed countries (with a Human Development Index greater than or equal to 0.8) and developing countries (with a Human Development Index less than 0.8).
# This line chart shows the average total COVID-19 cases per year for developed and developing countries, providing insights into how the pandemic has impacted healthcare infrastructure in both groups.

# In[15]:


# Create a derived column to categorize countries as developed or developing
df_yearly['country_category'] = df_yearly['human_development_index'].apply(lambda x: 'Developed' if x >= 0.8 else 'Developing')

# Calculate the average total cases and hospital beds per thousand for each year and country category
line_data = df_yearly.groupby(['year', 'country_category'])['hospital_beds_per_thousand', 'total_cases'].mean().reset_index()

# Line plot for Hypothesis 1: Healthcare infrastructure vs. total cases
plt.figure(figsize=(10, 6))
sns.lineplot(data=line_data, x='year', y='total_cases', hue='country_category', style='country_category', markers=True, dashes=False, palette='viridis')
plt.title('Healthcare Infrastructure vs. Total COVID-19 Cases (Line Chart)')
plt.xlabel('Year')
plt.ylabel('Total COVID-19 Cases (Averaged by Year)')
plt.show()


# Hypothesis 2: The COVID-19 pandemic has affected the availability of medical equipment and personnel in both developed and developing countries.
# 
# For Hypothesis 2, we can create a bar chart comparing the average total COVID-19 deaths per year against 'total_vaccinations_per_hundred','people_vaccinated_per_hundred'. taking human index to identify developed and developing countries from above code we check how the vaccination has impacted on total deaths
# 
# This bar chart shows the average total COVID-19 deaths per year for developed and developing countries, illustrating the relationship between medical equipment like vaccination and the pandemic's impact on these groups.

# In[16]:


# Calculate the average total_vaccinations_per_hundred and people_vaccinated_per_hundred for each year and country category
# to check the avaliabality of vacination impacted on deaths 
bar_data = df_yearly.groupby(['year', 'country_category'])['total_vaccinations_per_hundred','people_vaccinated_per_hundred', 'total_deaths'].mean().reset_index()

# Bar plot for Hypothesis 2: medical equipment vs. total deaths
plt.figure(figsize=(10, 6))
sns.barplot(data=bar_data, x='year', y='total_deaths', hue='country_category', palette='viridis')
plt.title('Medical equipment vs. Total COVID-19 Deaths (Bar Chart)')
plt.xlabel('Year')
plt.ylabel('Total COVID-19 Deaths (Averaged by Year)')
plt.show()


# Hypothesis 3: Healthcare systems in both developed and developing countries have responded to the pandemic, but their responses have been limited by the severity of the crisis.
# 
# Hypothesis 3 using a grouped bar chart to compare the average number of ICU beds per 100,000 people and the average number of COVID-19 deaths per million people in developed and developing countries.
# Calculate the average number of ICU beds per 100,000 people and the average number of COVID-19 deaths per million people in developed and developing countries:
# Create a grouped bar chart comparing the average number of ICU beds per 100,000 people and the average number of COVID-19 deaths per million people in developed and developing countries:
# This grouped bar chart compares the average number of ICU beds per 100,000 people and the average number of COVID-19 deaths per million people in developed and developing countries, offering insights into how healthcare systems in these countries have responded to the pandemic.

# In[18]:


bar_data = df_yearly.groupby('country_category')['icu_patients_per_million', 'new_deaths_per_million'].mean().reset_index()

bar_data = df_yearly.melt(id_vars=['country_category'], value_vars=['icu_patients_per_million', 'new_deaths_per_million'], var_name='metric', value_name='value')
bar_data['metric'] = bar_data['metric'].map({'icu_patients_per_million': 'ICU Patients per 100,000', 'new_deaths_per_million': 'COVID-19 Deaths per Million'})

plt.figure(figsize=(10, 6))
sns.barplot(data=bar_data, x='country_category', y='value', hue='metric', palette='viridis')
plt.title('ICU Beds per 100,000 and COVID-19 Deaths per Million in Developed and Developing Countries')
plt.xlabel('Country Category')
plt.ylabel('Value')
plt.legend(title='Metric')
plt.show()


# # 4. Modelling
# 
# Let's utilise a regression model to examine the association between healthcare infrastructure, health expenditure per person, the human development index, and the total number of COVID-19 instances based on the hypotheses given. For this, we'll employ the Python library scikit-learn.
#  importing the necessary functions and preprocess the data:

# In[24]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Remove rows with missing values
df_yearly_clean = df_yearly.dropna()

# Define features (X) and target variable (y)
X = df_yearly_clean[['hospital_beds_per_thousand', 'total_vaccinations_per_hundred', 'human_development_index']]
y = df_yearly_clean['total_cases']

# Log-transform the target variable to reduce the effect of extreme values
y = np.log1p(y)


# Split the data into training and testing sets, and fit a linear regression model:

# In[25]:


# Split the data into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions using the test data and evaluate the performance of the model
# You can tell how well the model is working by looking at the RMSE and R2 Score.

# In[26]:


# Make predictions using the test data
y_pred = model.predict(X_test)

# Calculate the root mean squared error (RMSE) and R-squared (R2) score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")


# You can see from these graphs how each feature relates to all of the COVID-19 cases as well as how effectively the model predicts the cases based on the features. The y-axis displays the log of all cases plus one because the target variable was log-transformed to lessen the impact of extreme values.

# In[27]:


# Create a function to plot actual values vs. predicted values for a single feature
def plot_actual_vs_predicted(feature_name, X_test, y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[feature_name], y_test, color='blue', label='Actual')
    plt.scatter(X_test[feature_name], y_pred, color='red', label='Predicted')
    plt.xlabel(feature_name)
    plt.ylabel('log(Total COVID-19 Cases + 1)')
    plt.title(f'Actual vs. Predicted Total COVID-19 Cases ({feature_name})')
    plt.legend()
    plt.show()

# Plot actual vs. predicted values for 'hospital_beds_per_thousand'
plot_actual_vs_predicted('hospital_beds_per_thousand', X_test, y_test, y_pred)
# Plot actual vs. predicted values for 'health_exp_per_capita'
plot_actual_vs_predicted('total_vaccinations_per_hundred', X_test, y_test, y_pred)

# Plot actual vs. predicted values for 'human_development_index'
plot_actual_vs_predicted('human_development_index', X_test, y_test, y_pred)

