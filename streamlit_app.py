import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load and preprocess datasets
@st.cache_data
def load_data():
    books_df = pd.read_csv('Books.csv', encoding='latin-1')
    ratings_df = pd.read_csv('Ratings.csv', encoding='latin-1')
    users_df = pd.read_csv('Users.csv', encoding='latin-1')
    
    # Deduplicate books
    books_df = books_df.drop_duplicates(subset=['Book-Title', 'Book-Author'], keep='first')
    
    return books_df, ratings_df, users_df

# Preprocess data
@st.cache_data
def preprocess_data(books_df):
    # Fill NaN values with empty strings
    books_df['Book-Title'] = books_df['Book-Title'].fillna('')
    books_df['Book-Author'] = books_df['Book-Author'].fillna('')
    books_df['Publisher'] = books_df['Publisher'].fillna('')
    
    # Create combined features
    books_df['combined_features'] = (
        books_df['Book-Title'] + ' ' + 
        books_df['Book-Author']
    ).str.lower()
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    
    # Remove any remaining NaN or empty strings
    valid_features = books_df['combined_features'].dropna().replace('', np.nan).dropna()
    
    # Fit and transform only valid features
    tfidf_matrix = tfidf.fit_transform(valid_features)
    
    # Reindex to match original dataframe
    valid_indices = valid_features.index
    books_df_filtered = books_df.loc[valid_indices].reset_index(drop=True)
    
    return books_df_filtered, tfidf_matrix, tfidf

# Fit NearestNeighbors model (cache to avoid recomputation)
@st.cache_resource
def fit_nearest_neighbors_model(_tfidf_matrix):
    nn_model = NearestNeighbors( algorithm='brute')
    nn_model.fit(_tfidf_matrix)
    return nn_model

# Recommendation function
def get_recommendations(book_title, books_df, tfidf_matrix, nn_model, top_n=20):
    # Find the index of the book
    book_indices = books_df[books_df['Book-Title'].str.lower().str.contains(book_title.lower(), case=False)].index
    
    if len(book_indices) == 0:
        return None
    
    # Use the first matching book
    book_index = book_indices[0]
    
    # Find similar books
    distances, indices = nn_model.kneighbors(tfidf_matrix[book_index], n_neighbors=top_n + 1)
    similar_indices = indices.flatten()[1:]
    
    recommended_books = books_df.iloc[similar_indices]
    
    return recommended_books

# Streamlit App
def main():
    st.header("\u2660 PageAI: The NLCSJ Book Recommendation Engine")
    st.markdown("Takes a book that user has read as input and gives recommendations. For now, we recommend that you skip the first 5 books due to the large frequency of duplicates. We are still refining our dataset")

    # Load and preprocess data
    books_df, ratings_df, users_df = load_data()
    books_df, tfidf_matrix, tfidf_vectorizer = preprocess_data(books_df)
    nn_model = fit_nearest_neighbors_model(tfidf_matrix)

    # Book input field in the center
    st.header("Find Your Next Read")
    book_title = st.text_input("Enter a book you've enjoyed (Press Enter to search)", "")

    # Search button
    if st.button('Get Recommendations'):
        if book_title.strip() == "":
            st.warning("Please enter a book title.")
        else:
            # Get recommendations
            recommendations = get_recommendations(book_title, books_df, tfidf_matrix, nn_model)
            
            if recommendations is not None and not recommendations.empty:
                st.subheader(f"Books Similar to {book_title}")
                
                # Display recommendations as a list
                for _, book in recommendations.iterrows():
                    st.write("----")
                    if 'Image-URL-S' in book and pd.notna(book['Image-URL-S']):
                        st.image(book['Image-URL-S'], width=100)
                    st.write(f"**{book['Book-Title']}**")
                    st.write(f"*by {book['Book-Author']}*")
                    if 'Publisher' in book and pd.notna(book['Publisher']):
                        st.write(f"Publisher: {book['Publisher']}")
            else:
                st.warning("No recommendations found. Please try another book.")

if __name__ == '__main__':
    main()
