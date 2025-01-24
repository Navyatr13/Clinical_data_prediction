import streamlit as st
import pandas as pd

st.title("Sepsis Prediction Dashboard")

uploaded_file = st.file_uploader("Upload patient data (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", data.head())
    st.write("Predictions will appear here after integration.")
