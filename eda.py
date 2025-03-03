import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("https://raw.github.com/Shubham-S151/Capstone_Project/main/Heart%20Disease%20Data.csv.gz")

st.title("Cardiovascular Disease Analysis")

# Display dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Missing Values")
st.write(df.isnull().sum())

# Pie Chart for Heart Disease Distribution
st.subheader("Heart Disease Distribution")
fig_pie = px.pie(df, names='Heart_Disease', title='Distribution of Heart Disease',
                 color_discrete_sequence=['lightgreen', 'red'])
st.plotly_chart(fig_pie)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
corr = df.select_dtypes(include=['number']).corr()
fig_heatmap = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index),
                                          colorscale='RdBu', showscale=True)
st.plotly_chart(fig_heatmap)

# KDE Plots for Numerical Features
st.subheader("KDE Plots for Numerical Features")
num_cols = df.select_dtypes(include=['number']).columns.tolist()
for col in num_cols:
    fig_kde = px.histogram(df, x=col, nbins=50, marginal='box', opacity=0.6,
                           title=f'Distribution of {col}', color_discrete_sequence=['blue'])
    st.plotly_chart(fig_kde)

# Box Plots for Outliers
st.subheader("Box Plots for Numerical Features")
for col in num_cols:
    fig_box = px.box(df, y=col, title=f'Boxplot of {col}', color_discrete_sequence=['orange'])
    st.plotly_chart(fig_box)

# Count Plots for Categorical Features
st.subheader("Categorical Feature Distribution")
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    fig_count = px.bar(df[col].value_counts().reset_index(), x='index', y=col,
                       title=f'Count Plot of {col}', color_discrete_sequence=['green'])
    st.plotly_chart(fig_count)

# Bivariate Analysis - Box Plots
st.subheader("Bivariate Analysis: Numerical Features vs Heart Disease")
for col in num_cols:
    fig_bivar = px.box(df, x='Heart_Disease', y=col, title=f'{col} vs Heart Disease')
    st.plotly_chart(fig_bivar)

# Bivariate Analysis - Categorical Features vs Heart Disease
st.subheader("Bivariate Analysis: Categorical Features vs Heart Disease")
for col in cat_cols:
    if col != 'Heart_Disease':
        cross_tab = df.groupby([col, 'Heart_Disease']).size().reset_index(name='count')
        fig_cat_bivar = px.bar(cross_tab, x=col, y='count', color='Heart_Disease',
                               title=f'{col} vs Heart Disease', barmode='group')
        st.plotly_chart(fig_cat_bivar)
