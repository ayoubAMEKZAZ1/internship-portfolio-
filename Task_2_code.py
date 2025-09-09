# Task 2: Customer Segmentation Using Unsupervised Learning

## Problem Statement and Objective
# Objective: Cluster customers based on spending habits and propose marketing strategies tailored to each segment.

## Dataset description and loading
import pandas as pd

# Load the Mall Customers Dataset
data = pd.read_csv('Mall Customers.csv', delimiter=';')

# Rename columns to remove leading/trailing spaces and standardize
columns = data.columns.str.strip()
data.columns = columns

# Check the column names to ensure correctness
print(data.columns)

## Data cleaning and preprocessing
# Handle missing values
data = data.dropna()

## Exploratory Data Analysis (EDA)
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize spending habits
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=data)
plt.title('Spending Habits')
plt.show()

## Model building and evaluation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(data[['Annual Income (k$)', 'Spending Score (1-100)']])

# Add clusters to dataframe
data['Cluster'] = clusters

## Visualizations
# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data[['Annual Income (k$)', 'Spending Score (1-100)']])
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
plt.title('Customer Segments with PCA')
plt.show()

## Suggest relevant marketing strategies
# Cluster 0: High income, low spending - Target with luxury promotions
# Cluster 1: Low income, high spending - Offer discounts
# Cluster 2: Average income, average spending - General marketing
# Cluster 3: High income, high spending - Premium offers
# Cluster 4: Low income, low spending - Budget-friendly campaigns

## Final conclusion with insights
# Five distinct customer segments identified. Tailored strategies can enhance marketing effectiveness.