
# Task 1: Exploring and Visualizing the Iris Dataset
# Introduction and Problem Statement
# This script addresses Task 1 of the DevelopersHub Corporation Data Science & Analytics Internship. The objective is to explore and visualize the Iris dataset to understand its structure and relationships between features, preparing it for potential classification tasks. The dataset is loaded directly from the seaborn library, equivalent to the UCI Iris dataset (https://archive.ics.uci.edu/dataset/53/iris).
#
# Dataset Understanding and Description
# The Iris dataset, accessed via `seaborn.load_dataset('iris')`, contains 150 samples of iris flowers with features: sepal_length, sepal_width, petal_length, petal_width, and species (target, with three classes: setosa, versicolor, virginica). Each feature is numerical except for species, which is categorical.
#
# Approach
# 1. Load the Iris dataset using seaborn.
# 2. Inspect dataset structure (shape, columns, data types, missing values).
# 3. Perform exploratory data analysis (EDA) with:
#    - Scatter plot (sepal length vs. sepal width).
#    - Histogram (petal length distribution).
#    - Box plot (sepal width by species).
#    - Pairplot with regression lines (innovative) to explore feature relationships.
# 4. Evaluate data quality and summarize findings.

# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set visualization style
sns.set_style('whitegrid')
# %matplotlib inline  # Note: This is a Jupyter-specific command; for .py, ensure plots are displayed with plt.show()

# Data Loading and Inspection
# Load the Iris dataset from seaborn and inspect its structure.

# Load dataset from seaborn
data = sns.load_dataset('iris')

# Display shape, columns, and first few rows
print('Dataset Shape:', data.shape)
print('\nColumn Names:', data.columns.tolist())
print('\nFirst 5 Rows:')
print(data.head())

# Check missing values
print('\nMissing Values:')
print(data.isnull().sum())

# Check data types
print('\nData Types:')
print(data.dtypes)

# Exploratory Data Analysis (EDA)
# Visualize feature distributions and relationships to understand patterns.

# Scatter plot: Sepal Length vs. Sepal Width
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='sepal_length', y='sepal_width', hue='species', style='species')
plt.title('Sepal Length vs. Sepal Width by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

# Histogram: Petal Length
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='petal_length', hue='species', multiple='stack', bins=20)
plt.title('Petal Length Distribution by Species')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Count')
plt.show()

# Box plot: Sepal Width by Species
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='species', y='sepal_width')
plt.title('Sepal Width Distribution by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Width (cm)')
plt.show()

# Pairplot with regression lines (innovative)
sns.pairplot(data, hue='species', diag_kind='kde', kind='reg')
plt.suptitle('Pairplot of Iris Features with Regression Lines', y=1.02)
plt.show()

# Evaluation Metrics
# Since Task 1 focuses on EDA, evaluation centers on data quality and insights:
# - Missing Values: None detected, confirming a clean dataset.
# - Feature Relationships: Strong correlations between petal_length and petal_width; setosa is distinctly separable.
# - Outliers: Minor outliers in sepal_width for virginica, as seen in box plots.

# Conclusion
# - Data Quality: The Iris dataset is clean with no missing values, suitable for classification tasks.
# - Insights: Setosa is easily separable from versicolor and virginica in petal measurements. Petal_length and petal_width show strong positive correlations, as highlighted by the pairplot with regression lines.
# - Innovative Approach: The pairplot with regression lines provides a clear visualization of feature relationships, enhancing understanding of species separation.
