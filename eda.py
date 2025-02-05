import streamlit as st
import pandas as pd
import requests
import io
import gzip
import seaborn as sns
import matplotlib.pyplot as plt

# App Title
st.title("📊 Heart Disease Data Analysis (Live from GitHub)")

# Hardcoded GitHub Raw File URL (No User Input Needed)
GITHUB_URL = "https://raw.github.com/Shubham-S151/Capstone_Project/main/Heart%20Disease%20Data.csv.gz"

# Function to Load Dataset
def load_data(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad requests
        
        # Validate that it's a proper gzip file
        if response.content[:2] != b'\x1f\x8b':  # Gzip files start with magic bytes 0x1f 0x8b
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
