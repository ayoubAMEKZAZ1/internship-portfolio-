# Task 1: Exploring and Visualizing the Iris Dataset

## Objective
Explore and visualize the Iris dataset to understand its structure and relationships between features, as part of the DevelopersHub Corporation Data Science & Analytics Internship. The dataset is loaded directly from the seaborn library, equivalent to the UCI Iris dataset (https://archive.ics.uci.edu/dataset/53/iris).

## Approach
1. Loaded the Iris dataset using `seaborn.load_dataset('iris')`.
2. Inspected dataset structure (shape, columns, data types, missing values).
3. Performed exploratory data analysis (EDA) with:
   - Scatter plot (sepal length vs. sepal width).
   - Histogram (petal length distribution).
   - Box plot (sepal width by species).
   - Pairplot with regression lines (innovative) to explore feature relationships.
4. Evaluated data quality and summarized findings.

## Results
- **Data Quality**: No missing values, confirming a clean dataset.
- **Feature Relationships**: Strong correlations between petal length and petal width; setosa is distinctly separable from versicolor and virginica.
- **Insights**: Petal measurements effectively distinguish species, with minor outliers in sepal width for virginica.

