import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast
import numpy as np

st.set_page_config(
    page_title="Intelligent Book Recommendation System",
    layout="wide"
)

#--------------------------------------------------------------
# Dataset Loading
#--------------------------------------------------------------

@st.cache_resource
def load_data():
    data = pd.read_csv("../datasets/cleaned_audible_catalog.csv")
    data['Genre_List'] = data['Genre_List'].apply(ast.literal_eval)
    return data

df = load_data()



tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


st.write("# Book Recommendationt System")
st.write("Welcome to the Intelligent Book Recommendation System! Explore personalized book suggestions based on your preferences.")
st.write("Whether you're looking for similar books or want to browse by genre, we've got you covered.")          
st.write("Start by selecting a book or genre to discover your next great read!")         

tab1, tab2 = st.tabs([
    "Content-Based Recommendations",
    "Genre-Based Recommendations"
])

with tab1:

    book_choice = st.selectbox(
        "Choose a book: ",
        df['Book_Name'].tolist(),
        index=None
    )

    if st.button("Get Recommendations"):
        if book_choice == None:
            st.warning("Please select a book to get recommendations.")
        else:
            matches = df[df['Book_Name'] == book_choice]
            if len(matches) > 0:
                idx = matches.index[0]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                book_indices = [i[0] for i in sim_scores[1:6]]
                st.write(f"### Book Recommendations similar to ***{book_choice}***")
                st.dataframe(
                    df.iloc[book_indices][['Book_Name','Author', 'Rating', 'Main_Genre']],
                    hide_index=True
                )
            else:
                st.warning("Book not found ")

with tab2:
    all_genres = sorted({genre for sublist in df['Genre_List'] for genre in sublist})
    selected_genre = st.selectbox(
        "Select a genre to browse books: ",
        all_genres,
        index=None
    )

    if st.button("Get Recommendations "):
            if selected_genre != 'None':
                mask = df['Genre_List'].apply(lambda x: selected_genre in x)
                genre_recommendations = df[mask].sort_values(by='Rating', ascending=False)
                
                if not genre_recommendations.empty:
                    st.write(f"##### Books categorized under ***{selected_genre}***:")
                    st.dataframe(
                        genre_recommendations[['Book_Name', 'Author', 'Rating', 'Main_Genre', 'Genre_List']], 
                        hide_index=True
                    )
                else:
                    st.info("No books found for this genre.")
            else:
                st.warning("Please select a genre")