import streamlit as st
import pandas as pd
from data.data import load_data
from sidebar.filters import filters, sidebar_filters
import plotly.express as px

st.set_page_config(
    page_title="Exploratory Data Analysis (EDA)"
)

#--------------------------------------------------------------
# Load Sidebar Filters
#--------------------------------------------------------------
with st.sidebar:
    sidebar_filters()

#--------------------------------------------------------------
# Dataset Loading
#--------------------------------------------------------------


df = load_data(filters)

if df is None or df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

#--------------------------------------------------------------
# Chart Functions
#--------------------------------------------------------------

def book_rating_distribution(df):
    st.write("### Book Rating Distribution")
    bins = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    labels = ['1-1.5', '1.5-2', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-4.5', '4.5-5']
    rating_bucket = pd.cut(df['Rating'], bins=bins, labels=labels, include_lowest=True)
    rating_bucket_counts = rating_bucket.value_counts().sort_index().reset_index(name='Count')
    st.bar_chart(
        data=rating_bucket_counts,
        x='Rating',
        y='Count'
    )

def top_genres(df):
    popular_genres = (
        df['Main_Genre']
        .value_counts()
        .reset_index(name='Count')
        .head(10)
    )

    st.write("### Top Genres")
    fig = px.bar(
        popular_genres,
        x="Main_Genre",
        y="Count",
        category_orders={'Main_Genre': popular_genres['Main_Genre'].tolist()},
    )
    st.plotly_chart(fig)


def popular_books(df):
    df_extended = df.copy()
    df_extended['Popularity'] = df_extended['Rating'] * df_extended['Number_of_Reviews']
    top_books = (
        df_extended
        .sort_values(by='Popularity', ascending=False)
        [['Book_Name', 'Author', 'Rating', 'Number_of_Reviews']]
    )
    return top_books


#--------------------------------------------------------------
# EDA
#--------------------------------------------------------------

st.write("# Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.metric(label="Total Books", value=len(df))
with col2:
    with st.container(border=True):
        st.metric(label='Popular Genre', value=df['Main_Genre'].mode()[0])

with st.container(border=True):
    st.write("### Popular Books")
    st.dataframe(popular_books(df))

with st.container(border=True):
    book_rating_distribution(df)

with st.container(border=True):
    top_genres(df)


