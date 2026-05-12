import streamlit as st
import pandas as pd

@st.cache_resource
def load_data(filter_dict):
    data = pd.read_csv("data/cleaned_audible_catalog.csv")

    for key, value in filter_dict.items():
        if value is not None:
            if isinstance(value, tuple):
                data = data[(data[key] >= value[0]) & (data[key] <= value[1])]

            else:
                data = data[data[key] == value]

    return data