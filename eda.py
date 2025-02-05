import streamlit as st
import pandas as pd
import requests
import io
import gzip
import seaborn as sns
import matplotlib.pyplot as plt

# App Title
st.title("📊 Heart Disease Data Analysis (Live from GitHub)")

# Hardcoded GitHub Raw File URL
GITHUB_URL = "https://raw.github.com/Shubham-S151/Capstone_Project/main/Heart%20Disease%20Data.csv.gz"

# Function to Load Dataset
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
    st.write(df.head())

    st.write("### 📊 Summary Statistics")
    st.write(df.describe())

    st.write("### ❗ Missing Values")
    st.write(df.isnull().sum())

    # Visualization Options
    st.sidebar.header("📊 Visualization Options")

    # Select Visualization Type
    vis_option = st.sidebar.selectbox("Choose a Visualization", ["Histogram", "Boxplot", "Scatter Plot", "Correlation Heatmap", "Pairplot", "Bar Chart"])

    # Histogram
    if vis_option == "Histogram":
        st.write("### 📈 Data Distribution")
        num_cols = df.select_dtypes(include=["number"]).columns
        selected_col = st.selectbox("Select a column for distribution plot:", num_cols)
        
        if selected_col:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[selected_col], kde=True, bins=30, ax=ax)
            st.pyplot(fig)

    # Boxplot
    elif vis_option == "Boxplot":
        st.write("### 📦 Boxplot (Detect Outliers)")
        num_cols = df.select_dtypes(include=["number"]).columns
        selected_col = st.selectbox("Select a column for boxplot:", num_cols)

        if selected_col:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(y=df[selected_col], ax=ax)
            st.pyplot(fig)

    # Scatter Plot
    elif vis_option == "Scatter Plot":
        st.write("### 🔎 Scatter Plot (Relationships between variables)")
        num_cols = df.select_dtypes(include=["number"]).columns
        col_x = st.selectbox("Select X-axis variable:", num_cols)
        col_y = st.selectbox("Select Y-axis variable:", num_cols)

        if col_x and col_y:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(x=df[col_x], y=df[col_y], ax=ax)
            st.pyplot(fig)

    # Correlation Heatmap
    elif vis_option == "Correlation Heatmap":
        st.write("### 🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Pairplot
    elif vis_option == "Pairplot":
        st.write("### 🔗 Pairplot (Variable Relationships)")
        selected_cols = st.multiselect("Select up to 5 columns:", df.select_dtypes(include=["number"]).columns)
        
        if selected_cols and len(selected_cols) <= 5:
            fig = sns.pairplot(df[selected_cols],markers='+)
            st.pyplot(fig)

    # Bar Chart for Categorical Variables
    elif vis_option == "Bar Chart":
        st.write("### 📊 Bar Chart (Categorical Variables)")
        cat_cols = df.select_dtypes(include=["object"]).columns

        if len(cat_cols) > 0:
            selected_col = st.selectbox("Select a categorical column:", cat_cols)

            if selected_col:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(x=df[selected_col], ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.warning("No categorical columns available in this dataset.")
