import streamlit as st
import pandas as pd
import requests
import io
import gzip
import plotly.express as px
import plotly.figure_factory as ff

# App Title
st.title("📊 Heart Disease Data Analysis (Live from GitHub)")

# Hardcoded GitHub Raw File URL
GITHUB_URL = "https://raw.github.com/Shubham-S151/Capstone_Project/main/Heart%20Disease%20Data.csv.gz"

# Function to Load Dataset
@st.cache_data
def load_data(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad requests

        # Validate that it's a proper gzip file
        if response.content[:2] != b'\x1f\x8b':  
            raise ValueError("Invalid GZ file. Please check the GitHub URL.")

        # Decompress and Read CSV
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            df = pd.read_csv(f)
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"🚨 Failed to fetch data from GitHub: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

# Load Data
df = load_data(GITHUB_URL)

# Display Data if Successfully Loaded
if df is not None:
    st.write("### 🔍 Dataset Preview")
    st.dataframe(df.head())

    st.write("### 📊 Summary Statistics")
    st.write(df.describe())

    st.write("### ❗ Missing Values")
    st.write(df.isnull().sum())

    # Visualization Options
    st.sidebar.header("📊 Visualization Options")

    # Select Visualization Type
    vis_option = st.sidebar.selectbox("Choose Analysis Type:", 
                                      ["Select an option", "Descriptive Analysis", "Univariate Analysis", 
                                       "Bivariate Analysis", "Multivariate Analysis"])

    # Descriptive Analysis
    if vis_option == "Descriptive Analysis":
        st.write("### 📈 Data Distribution")
        num_cols = df.select_dtypes(include=["number"]).columns
        selected_col = st.selectbox("Select a column:", num_cols)

        if selected_col:
            fig = px.histogram(df, x=selected_col, nbins=30, marginal="box", title=f"Distribution of {selected_col}")
            st.plotly_chart(fig)

    # Univariate Analysis - Boxplot
    elif vis_option == "Univariate Analysis":
        st.write("### 📦 Boxplot (Outliers Detection)")
        num_cols = df.select_dtypes(include=["number"]).columns
        selected_col = st.selectbox("Select a column:", num_cols)

        if selected_col:
            fig = px.box(df, y=selected_col, title=f"Boxplot of {selected_col}")
            st.plotly_chart(fig)

    # Bivariate Analysis - Scatter Plot
    elif vis_option == "Bivariate Analysis":
        st.write("### 🔎 Scatter Plot (Variable Relationships)")
        num_cols = df.select_dtypes(include=["number"]).columns
        col_x = st.selectbox("Select X-axis variable:", num_cols)
        col_y = st.selectbox("Select Y-axis variable:", num_cols)

        if col_x and col_y:
            fig = px.scatter(df, x=col_x, y=col_y, title=f"Scatter Plot: {col_x} vs {col_y}")
            st.plotly_chart(fig)

    # Multivariate Analysis - Correlation Heatmap
    elif vis_option == "Multivariate Analysis":
        st.write("### 🔥 Correlation Heatmap")
        corr_matrix = df.corr()

        fig = px.imshow(corr_matrix, text_auto=True, labels=dict(color="Correlation"),
                        title="Correlation Matrix Heatmap", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig)

    # Categorical Variable Analysis - Bar Chart
    st.sidebar.subheader("📊 Categorical Variable Analysis")
    cat_cols = df.select_dtypes(include=["object"]).columns

    if len(cat_cols) > 0:
        selected_cat_col = st.sidebar.selectbox("Select a categorical column:", cat_cols)

        if selected_cat_col:
            st.write(f"### 📊 Bar Chart of {selected_cat_col}")
            fig = px.bar(df[selected_cat_col].value_counts().reset_index(), 
                         x="index", y=selected_cat_col, 
                         labels={"index": selected_cat_col, selected_cat_col: "Count"},
                         title=f"Bar Chart of {selected_cat_col}")
            st.plotly_chart(fig)
    else:
        st.sidebar.warning("No categorical columns available in this dataset.")
