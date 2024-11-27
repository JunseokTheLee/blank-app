import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess datasets
@st.cache_data
def load_data():
    # Load datasets
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
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Remove any remaining NaN or empty strings
    valid_features = books_df['combined_features'].dropna().replace('', np.nan).dropna()
    
    # Fit and transform only valid features
    tfidf_matrix = tfidf.fit_transform(valid_features)
    
    # Reindex to match original dataframe
    valid_indices = valid_features.index
    books_df_filtered = books_df.loc[valid_indices].reset_index(drop=True)
    
    return books_df_filtered, tfidf_matrix, tfidf

# Recommendation function
def get_recommendations(book_title, books_df, tfidf_matrix, tfidf_vectorizer, top_n=10):
    # Find the index of the book
    book_indices = books_df[books_df['Book-Title'].str.lower().str.contains(book_title.lower(), case=False)].index
    
    if len(book_indices) == 0:
        return None
    
    # Use the first matching book
    book_index = book_indices[0]
    
    # Get cosine similarity for the selected book
    book_vector = tfidf_vectorizer.transform([books_df.loc[book_index, 'combined_features']])
    cosine_sim = cosine_similarity(book_vector, tfidf_matrix).flatten()
    
    # Get top recommendations, excluding the input book
    similar_indices = cosine_sim.argsort()[::-1]
    similar_indices = [idx for idx in similar_indices if idx != book_index][:top_n]
    
    recommended_books = books_df.iloc[similar_indices]
    
    return recommended_books

# Streamlit App
def main():
    st.header("PageAI: The NLCSJ Book Recommendation Engine")
    st.markdown("Takes a book that user has read as input and gives recommendations")

    # Load and preprocess data
    books_df, ratings_df, users_df = load_data()
    books_df, tfidf_matrix, tfidf_vectorizer = preprocess_data(books_df)

    # Book selection
    st.sidebar.header("Find Your Next Read")
    selected_book = st.sidebar.selectbox(
        "Select a book you've enjoyed", 
        sorted(books_df['Book-Title'].unique())
    )

    if st.sidebar.button('Get Recommendations'):
        # Get recommendations
        recommendations = get_recommendations(selected_book, books_df, tfidf_matrix, tfidf_vectorizer)
        
        if recommendations is not None and not recommendations.empty:
            st.subheader(f"Books Similar to {selected_book}")
            
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
