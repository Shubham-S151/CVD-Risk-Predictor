import streamlit as st
import pandas as pd
import requests
import io
import zipfile
import gzip
import seaborn as sns
import matplotlib.pyplot as plt

# App Title
st.title("Exploratory Data Analysis")

# Input for GitHub raw file URL
github_url = st.text_input("https://github.com/Shubham-S151/Capstone_Project/blob/main/compressed_data.csv.gz")

# Detect File Type
def detect_file_type(url):
    if url.endswith(".csv"):
        return "csv"
    elif url.endswith(".zip"):
        return "zip"
    elif url.endswith(".gz"):
        return "gz"
    else:
        return None

# Load Dataset
def load_data(url, file_type):
    response = requests.get(url)
    response.raise_for_status()  # Raise error for invalid requests

    if file_type == "csv":
        return pd.read_csv(io.StringIO(response.text))
    
    elif file_type == "zip":
        with zipfile.ZipFile(io.BytesIO(response.content), "r") as z:
            file_names = z.namelist()  # Get files inside ZIP
            csv_files = [f for f in file_names if f.endswith(".csv")]
            if not csv_files:
                raise ValueError("No CSV found in ZIP")
            with z.open(csv_files[0]) as f:
                return pd.read_csv(f)
    
    elif file_type == "gz":
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            return pd.read_csv(f)

    else:
        return None

# Process and Display Data
if github_url:
    file_type = detect_file_type(github_url)
    
    if file_type:
        try:
            df = load_data(github_url, file_type)
            
            st.write("### 🔍 Dataset Preview")
            st.write(df.head())

            st.write("### 📊 Summary Statistics")
            st.write(df.describe())

            st.write("### ❗ Missing Values")
            st.write(df.isnull().sum())

            # Visualization
            if not df.select_dtypes(include=["number"]).empty:
                st.write("### 📈 Data Distribution")
                num_cols = df.select_dtypes(include=["number"]).columns
                
                # Select column for visualization
                selected_col = st.selectbox("Select a numeric column for distribution plot:", num_cols)
                
                if selected_col:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(df[selected_col], kde=True, bins=30, ax=ax)
                    st.pyplot(fig)

            # Correlation Heatmap
            if len(df.select_dtypes(include=["number"]).columns) > 1:
                st.write("### 🔥 Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.warning("⚠️ Unsupported file type. Please upload a CSV, ZIP (containing CSV), or GZ file.")

# Run with: `streamlit run app.py`
