import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, movies_path, ratings_path):
        self.dataset_movies = pd.read_csv(movies_path)
        self.dataset_ratings = pd.read_csv(ratings_path)

    def get_dataset_stats(self):
        n_userID = len(self.dataset_ratings['userId'].unique())
        n_movieID = len(self.dataset_ratings['movieId'].unique())
        n_rating = len(self.dataset_ratings)

        stats = {
            "total_ratings": n_rating,
            "unique_users": n_userID,
            "unique_movies": n_movieID,
            "avg_ratings_per_user": round(n_rating / n_userID),
            "avg_ratings_per_movie": round(n_rating / n_movieID)
        }
        return stats

    def get_movie_rating_stats(self):
        mean_rating = self.dataset_ratings.groupby('movieId')[['rating']].mean()
        
        movie_stats = self.dataset_ratings.groupby('movieId')[['rating']].agg(['count', 'mean'])
        movie_stats.columns = movie_stats.columns.droplevel()
        
        return {
            "lowest_rated_movie": mean_rating['rating'].idxmin(),
            "highest_rated_movie": mean_rating['rating'].idxmax(),
            "movie_rating_stats": movie_stats
        }

    def get_user_rating_distribution(self):
        return self.dataset_ratings[['userId', 'rating']].groupby('userId').count().reset_index()
