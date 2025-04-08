import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Sample dataset
movies = {
    "Title": ["Inception", "Interstellar", "The Dark Knight", "Memento", "Tenet"],
    "Description": [
        "A thief who enters the dreams of others to steal secrets.",
        "A team travels through a wormhole in space.",
        "Batman fights crime in Gotham City.",
        "A man with short-term memory loss investigates his wife's murder.",
        "Time manipulation and espionage."
    ]
}

df = pd.DataFrame(movies)
print(df)
# Convert text descriptions into TF-IDF feature vectors
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Description"])
# Compute cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)


def recommend_movies(movie_title, df, similarity_matrix):
    if movie_title not in df["Title"].values:
        return "Movie not found in dataset."

    # Find index of the given movie
    movie_idx = df[df["Title"] == movie_title].index[0]
    #print(movie_idx)
    # Get similarity scores for all movies
    scores = list(enumerate(similarity_matrix[movie_idx]))
    #print(scores)
    # Sort movies by similarity score
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    #print(sorted_scores)
    # Get top 3 similar movies (excluding the given movie)
    recommended_movies = [df.iloc[i[0]]["Title"] for i in sorted_scores[1:4]]

    return recommended_movies
movie_name = "The Dark Knight"
recommendations = recommend_movies(movie_name, df, similarity_matrix)
print(f"Movies similar to '{movie_name}': {recommendations}")
