import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from pathlib import Path

# Cache data loading and processing
@st.cache_data
def load_and_process_data():
    """Load and process movie data with error handling"""
    try:
        # Load the data
        data_path = Path(__file__).parent / 'data'
        movies = pd.read_csv(data_path / 'movies.csv')
        ratings = pd.read_csv(data_path / 'ratings.csv')
        
        # Merge datasets
        df = ratings.merge(movies, on='movieId')
        df = df[['userId', 'title', 'rating']]
        
        # Create user-item matrix (avoid inplace warning)
        user_movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')
        user_movie_matrix = user_movie_matrix.fillna(0)
        
        # Compute movie similarity
        movie_similarity = cosine_similarity(user_movie_matrix.T)
        movie_similarity_df = pd.DataFrame(
            movie_similarity, 
            index=user_movie_matrix.columns,
            columns=user_movie_matrix.columns
        )
        
        return movies, movie_similarity_df
    
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Load data
movies, movie_similarity_df = load_and_process_data()

def recommend_movies(movie_name, top_n=5):
    """Get movie recommendations based on similarity scores"""
    if movie_name not in movie_similarity_df.columns:
        return None, f"Movie '{movie_name}' not found in dataset."
    
    similar_scores = movie_similarity_df[movie_name].sort_values(ascending=False)[1:top_n+1]
    return similar_scores, None

# Streamlit UI
st.title("🎬 Movie Recommender System")
st.write("Find movies similar to your favorites!")

# Add some statistics
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Movies", len(movies))
with col2:
    st.metric("Movies in Matrix", len(movie_similarity_df.columns))

# Get list of all movies for dropdown (sorted for better UX)
movie_list = sorted(movies['title'].tolist())

# Create dropdown for movie selection with search
selected_movie = st.selectbox(
    "Select a movie you like:",
    movie_list,
    help="Start typing to search for a movie"
)

# Number of recommendations slider
num_recommendations = st.slider(
    "Number of recommendations:",
    min_value=1,
    max_value=20,
    value=5
)

if st.button("Get Recommendations", type="primary"):
    with st.spinner("Finding similar movies..."):
        recommendations, error = recommend_movies(selected_movie, num_recommendations)
        
        if error:
            st.error(error)
        elif recommendations is not None and len(recommendations) > 0:
            st.success(f"Found {len(recommendations)} recommendations!")
            st.write("### 🍿 Movies you might like:")
            
            # Display as a nice table
            rec_df = pd.DataFrame({
                'Movie Title': recommendations.index,
                'Similarity Score': recommendations.values
            })
            rec_df['Similarity %'] = (rec_df['Similarity Score'] * 100).round(2)
            rec_df.index = range(1, len(rec_df) + 1)
            
            st.dataframe(
                rec_df[['Movie Title', 'Similarity %']], 
                use_container_width=True,
                hide_index=False
            )
        else:
            st.warning("No recommendations found for this movie.")

# Add footer
st.divider()
st.caption("Built with Streamlit • Data-driven movie recommendations using collaborative filtering")