# streamlit_app.py
# Streamlit-based options flow summarizer.

import streamlit as st
import pandas as pd

st.title("Options Flow Summarizer")
st.write("Upload a CSV to analyze options flow.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of data:")
    st.dataframe(df.head())
    st.success("CSV uploaded successfully!")