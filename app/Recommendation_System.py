import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

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
    return data

df = load_data()



tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


st.write("# Book Recommendationt System")


tab1, tab2 = st.tabs([
    "Content-Based Recommendations",
    "Genre-Based Recommendations"
])

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
