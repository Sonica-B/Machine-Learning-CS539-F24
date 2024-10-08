#data manipulation
import pandas as pd
import numpy as np

#data visualizations
import matplotlib.pyplot as plt
import seaborn as sns

#use sklearn to import a dataset

#load dataset
df = pd.read_csv(r"D:\WPI Assignments\Machine Learning CS539-F24\Homeworks\Homework1\datasets\earth_surface_temperatures.csv")
df.head()

#To get highlevel info about the dataset
df.info()
#To get details of dataset
df.describe()
#Check for missing values in all columns
df.isnull().sum()

# Fill missing values with the mean temperature
df['Temperature'] = df['Temperature'].fillna(df['Temperature'].mean())

#Using Interpolate to estimate the missing values based on surrounding data.
df['Monthly_variation'] = df['Monthly_variation'].interpolate(method='linear')
df['Anomaly'] = df['Anomaly'].interpolate(method='linear')

#Check
df.isnull().sum()

# Combine Year and Month into a single Date column
df['Date'] = pd.to_datetime(df[['Years', 'Month']].assign(DAY=1))


#check
print(df['Date'].dtype)

# Formating Date as MM-YYYY
#df['Date'] = df['Date'].dt.strftime('%m-%Y')

# Use IQR to detect outliers
Q1 = df['Temperature'].quantile(0.25)
Q3 = df['Temperature'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Temperature'] < (Q1 - 1.5 * IQR)) | (df['Temperature'] > (Q3 + 1.5 * IQR))]

# Create a box plot for temperature across different years
plt.figure(figsize=(120, 60))
sns.boxplot(x='Country', y='Temperature', data=df)
plt.title('Box Plot of Temperatures wrt country')
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
plt.show()


# Summary statistics
summary_stats = ['Temperature', 'Monthly_variation', 'Anomaly']
print(df[summary_stats].mean)
print(df[summary_stats].median)
print(df[summary_stats].std)

country_avg_temp = df.groupby('Country')['Temperature'].mean()
print(country_avg_temp)

# Plot global temperature trend
global_avg_temp = df.groupby('Years')['Temperature'].mean()
plt.plot(global_avg_temp)
plt.title('Global Temperature Trend Over Time')
plt.xlabel('Years')
plt.ylabel('Average Temperature')
plt.show()

# Group by country and month to find highest and lowest temperatures
monthly_temps = df.groupby(['Country', 'Month'])['Temperature'].agg(['max', 'min'])
print(monthly_temps)

monthly_anomalies = df.groupby('Month')['Anomaly'].mean()
print(monthly_anomalies)

# Pivot table to show anomalies by month and year
anomaly_pivot = df.pivot_table(values='Anomaly', index='Years', columns='Month', aggfunc='mean')

# Plot heatmap
sns.heatmap(anomaly_pivot, center=0)

countries = ['USA', 'India', 'Brazil', 'Russia', 'Australia']
df_five_countries = df[df['Country'].isin(countries)]

# Plot temperature trends for the five countries
sns.lineplot(x='Years', y='Temperature', hue='Country', data=df_five_countries)

# Calculate correlation
correlation = df[['Temperature', 'Anomaly']].corr()

# Scatter plot to visualize correlation
#sns.scatterplot(x='Temperature', y='Anomaly',style= , data=df)
sns.scatterplot(x='Temperature', y='Anomaly', hue='Month', style='Monthly_variation', data=df)


#Plotting Heatmap
#numeric_data = df.select_dtypes(include=['float64', 'int64'])
#correlation_matrix = df.corr()
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True)
plt.rcParams['figure.figsize'] = (20,7)
plt.title('Heatmap of Dataset')
plt.show()

#Plotting Combined Histogram for all columns
numeric_data.hist(bins=30, figsize=(12, 10), layout=(len(numeric_data.columns), 1))
plt.suptitle('Histograms of Numeric Variables', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Plotting individual Histograms
# Temperature
df['Temperature'].hist(bins=30, figsize=(8, 6))
plt.title('Histogram of Temperature')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.show()

# Date
df['Date'].hist(bins=30, figsize=(8, 6))
plt.title('Histogram of Date')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.show()

# Monthly_variation
df['Monthly_variation'].hist(bins=30, figsize=(8, 6))
plt.title('Histogram of Monthly_variation')
plt.xlabel('Monthly_variation')
plt.ylabel('Frequency')
plt.show()

# Anomaly
df['Anomaly'].hist(bins=30, figsize=(8, 6))
plt.title('Histogram of Anomaly')
plt.xlabel('Anomaly')
plt.ylabel('Frequency')
plt.show()


