import streamlit as st
import pandas as pd
import requests
import io
import seaborn as sns
import matplotlib.pyplot as plt

# App Title
st.title("📊 Load and Visualize Dataset from GitHub")

# Input for GitHub raw file URL
github_url = st.text_input("https://github.com/Shubham-S151/Capstone_Project/blob/main/compressed_data.csv.gz")

# Detect File Type
def detect_file_type(url):
    if url.endswith(".csv"):
        return "csv"
    elif url.endswith(".xlsx"):
        return "excel"
    elif url.endswith(".json"):
        return "json"
    else:
        return None

# Load Dataset
def load_data(url, file_type):
    response = requests.get(url)
    response.raise_for_status()  # Raise error for invalid requests

    if file_type == "csv":
        return pd.read_csv(io.StringIO(response.text))
    elif file_type == "excel":
        return pd.read_excel(io.BytesIO(response.content))
    elif file_type == "json":
        return pd.read_json(io.StringIO(response.text))
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
        st.warning("⚠️ Unsupported file type. Please upload a CSV, Excel, or JSON file.")

# Run with: `streamlit run app.py`
