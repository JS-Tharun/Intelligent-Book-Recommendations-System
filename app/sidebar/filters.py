import streamlit as st
import pandas as pd


filters = {}

def sidebar_filters():
    def filter_load_data():
        data = pd.read_csv("data/cleaned_audible_catalog.csv")
        return data
    df = filter_load_data()

    st.write("# Filters")
    rating_filter()
    author_filter(df)

def rating_filter():
    selected_rating = st.slider(
        "Ratings",
        min_value=1.0,
        max_value=5.0,
        value=(1.0, 5.0),
        step=0.5
    )
    filters['Rating'] = selected_rating

def author_filter(df):
    authors = df['Author'].unique().tolist()
    selected_authors = st.selectbox(
        "Author Name",
        options = authors,
        index=None
    )
    filters['Author'] = selected_authors