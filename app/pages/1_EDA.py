import streamlit as st
import pandas as pd

@st.cache_resource
def load_data():
    data = pd.read_csv("../datasets/cleaned_audible_catalog.csv")
    return data

df = load_data()



st.write("# Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.metric(label="Total Books", value=len(df))
with col2:
    with st.container(border=True):
        st.metric(label='Popular Genre', value=df['Main_Genre'].mode()[0])

st.dataframe(df.head())

