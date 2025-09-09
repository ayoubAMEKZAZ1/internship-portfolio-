# Task 5: Interactive Business Dashboard in Streamlit

## Problem Statement and Objective
# Objective: Develop an interactive dashboard for analyzing sales, profit, and segment-wise performance.

## Dataset description and loading
import pandas as pd
import streamlit as st

# Load the Global Superstore Dataset with proper encoding to handle special characters
data = pd.read_csv('Global_Superstore2.csv', encoding='latin1')
print(data.head())

## Data cleaning and preprocessing
# Handle missing values
data = data.dropna()

## Build a Streamlit dashboard with filters
st.title("Global Superstore Dashboard")
region = st.selectbox("Select Region", data['Region'].unique())
category = st.selectbox("Select Category", data['Category'].unique())
sub_category = st.selectbox("Select Sub-Category", data['Sub-Category'].unique())

filtered_data = data[(data['Region'] == region) & (data['Category'] == category) & (data['Sub-Category'] == sub_category)]

## Display key performance indicators (KPIs) using charts
import matplotlib.pyplot as plt

# Total Sales
st.subheader("Total Sales")
st.bar_chart(filtered_data['Sales'])

# Profit
st.subheader("Profit")
st.bar_chart(filtered_data['Profit'])

# Top 5 Customers by Sales
top_customers = filtered_data.groupby('Customer Name')['Sales'].sum().nlargest(5)
st.subheader("Top 5 Customers by Sales")
st.bar_chart(top_customers)

## Final conclusion with insights
# The dashboard allows interactive analysis of sales and profit by region, category, and sub-category.