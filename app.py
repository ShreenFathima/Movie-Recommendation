import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie data
movies_data = pd.read_csv('F:\Movie Recommendation\Movie Recommendation\Dataset\movies.csv')

# Preprocessing
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Create feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

# Streamlit application
st.title('Futuristic Movie Recommendation System')
st.markdown("<style>body { background-color: #282c34; color: white; }</style>", unsafe_allow_html=True)

movie_name = st.text_input('Enter your favorite movie name:', placeholder='e.g., Inception', key='movie_input')
num_recommendations = st.slider('Select the number of recommendations:', 1, 30, 10, key='num_recommendations')


if st.button('Get Recommendations'):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        st.write('Here are some movie suggestions for you:')
        
        for i, movie in enumerate(sorted_similar_movies):
            if i < num_recommendations:
                index = movie[0]
                title_from_index = movies_data[movies_data.index == index]['title'].values[0]
                st.write(f"{i + 1}. {title_from_index}")
    else:
        st.error("Movie not found. Please try another title.")
