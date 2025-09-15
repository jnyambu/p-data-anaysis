# Diabetes Dataset Analysis Assignment
# Data Analysis with Pandas and Visualization with Matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
# load  the datasest
df = pd.read_csv(r"C:\Users\NDICHU\Downloads\diabetes kaggle\diabeties\diabetes_dataset.csv")

# Display the first few rows to inspect the data
print("First 5 rows of the dataset:")
print(df.head())
print("\n" + "=" * 50)

# Display basic information about the dataset
print("Dataset Information:")

# Check for missing values in each column
print("Missing values per column:")
print(df.isnull().sum())
print("\n" + "=" * 50)
print("Missing values after replacing zeros with NaN:")
print(df.isnull().sum())
print("\n" + "=" * 50)
# Clean the dataset by filling missing values.
# For simplicity, we'll fill with the median of each column.
# The median is less sensitive to outliers than the mean.
for col in ['Insulin', 'BMI', 'Glucose', 'BloodPressure']:
    
    df[col].replace(0, np.nan, inplace=True)
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)

print("Missing values after imputation:")
print(df.isnull().sum())
print("\n" + "=" * 50)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Users/NDICHU/Downloads/diabetes kaggle/diabeties/diabetes_dataset.csv")
# Create a dummy DataFrame for demonstration data = {'species': ['setosa', 'setosa', 'versicolor', 'versicolor', 'virginica', 'virginica'],
data = {
    'petal_length': [1.4, 1.3, 4.5, 4.7, 6.0, 5.8],
    'petal_width': [0.2, 0.2, 1.5, 1.4, 2.2, 2.1],
    'species': ['setosa', 'setosa', 'versicolor', 'versicolor', 'virginica', 'virginica']
}

df = pd.DataFrame(data)
print(df)
# Display the first few rows of the DataFrame
print("First 5 rows of the DataFrame:")
print(df.head())
# Get a summary of the DataFrame
print("\nDataFrame Info:")
df.info()
# Compute basic statistics for numerical columns using .describe()
print("\nBasic statistics of numerical columns:")
print(df.describe())

# Group by a categorical column and compute the mean of a numerical column
# For this example, we'll group by 'species' and calculate the mean of 'petal_length'
print("\nMean 'petal_length' grouped by 'species':")
grouped_data = df.groupby('species')['petal_length'].mean()
print(grouped_data)

# Create a histogram to visualize the distribution of a numerical column
plt.figure(figsize=(8, 6))
plt.hist(df['petal_length'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Create a bar chart to visualize the grouped data
plt.figure(figsize=(8, 6))
grouped_data.plot(kind='bar', color=['lightgreen', 'salmon', 'cornflowerblue'])
plt.title('Mean Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Mean Petal Length (cm)')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.75)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
# 1. Create a sample time-series dataset
data = {'date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01']),
        'sales': [150, 165, 180, 175, 190, 210]}
sales_df = pd.DataFrame(data)

# 2. Create the line chart
plt.figure(figsize=(10, 6))
plt.plot(sales_df['date'], sales_df['sales'], marker='o', linestyle='-', color='b')
plt.title('Monthly Sales Trend', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# 1. Create a sample categorical dataset
data = {'species': ['A', 'B', 'C', 'A', 'B', 'C'],
        'petal_length': [3.1, 4.5, 5.2, 3.3, 4.7, 5.4]}
df_species = pd.DataFrame(data)

# 2. Group by species and calculate the mean
avg_petal_length = df_species.groupby('species')['petal_length'].mean()

# 3. Create the bar chart
plt.figure(figsize=(8, 6))
avg_petal_length.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Average Petal Length per Species', fontsize=16)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Average Petal Length (cm)', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Create a sample numerical dataset
np.random.seed(42)
data = {'sepal_width': np.random.normal(3.5, 0.7, 100)} # Normal distribution
df_sepal = pd.DataFrame(data)

# 2. Create the histogram
plt.figure(figsize=(8, 6))
plt.hist(df_sepal['sepal_width'], bins=15, color='purple', edgecolor='black', alpha=0.7)
plt.title('Distribution of Sepal Width', fontsize=16)
plt.xlabel('Sepal Width (cm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


