import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

class MovieRecommender:
    def __init__(self, movies_path, ratings_path):
        """
        Initialize the movie recommender with dataset paths
        
        :param movies_path: Path to movies CSV file
        :param ratings_path: Path to ratings CSV file
        """
        self.dataset_movies = pd.read_csv(movies_path)
        self.dataset_ratings = pd.read_csv(ratings_path)
        self.movie_titles = dict(zip(self.dataset_movies['movieId'], self.dataset_movies['title']))
        
        # Create sparse matrix
        self.X, self.user_map, self.movie_map, self.user_inv, self.movie_inv = self._create_matrix()
    
    def _create_matrix(self):
        """
        Create sparse matrix representation of ratings
        
        :return: Sparse matrix and mapping dictionaries
        """
        n = len(self.dataset_ratings['userId'].unique())
        m = len(self.dataset_ratings['movieId'].unique())
        
        user_map = dict(zip(np.unique(self.dataset_ratings['userId']), list(range(n))))
        movie_map = dict(zip(np.unique(self.dataset_ratings['movieId']), list(range(m))))
        
        user_inv = dict(zip(list(range(n)), np.unique(self.dataset_ratings['userId'])))
        movie_inv = dict(zip(list(range(m)), np.unique(self.dataset_ratings['movieId'])))
        
        user_index = [user_map[i] for i in self.dataset_ratings['userId']]
        movie_index = [movie_map[i] for i in self.dataset_ratings['movieId']]
        
        X = csr_matrix((self.dataset_ratings['rating'], (movie_index, user_index)), shape=(m, n))
        
        return X, user_map, movie_map, user_inv, movie_inv
    
    def find_similar_movies(self, movie_id, k=10, metric='cosine'):
        """
        Find similar movies based on collaborative filtering
        
        :param movie_id: ID of the reference movie
        :param k: Number of similar movies to return
        :param metric: Distance metric for similarity
        :return: List of similar movie IDs
        """
        try:
            movie_ind = self.movie_map[movie_id]
            movie_vec = self.X[movie_ind].reshape(1, -1)
            
            knn = NearestNeighbors(n_neighbors=k+1, algorithm='brute', metric=metric)
            knn.fit(self.X)
            
            _, indices = knn.kneighbors(movie_vec)
            
            # Exclude the movie itself and map back to original movie IDs
            similar_movie_indices = indices[0][1:]
            similar_movie_ids = [self.movie_inv[idx] for idx in similar_movie_indices]
            
            return similar_movie_ids
        
        except KeyError:
            print(f"Movie ID {movie_id} not found in the dataset.")
            return []
    
    def recommend_for_user(self, user_id, k=10):
        """
        Recommend movies for a specific user
        
        :param user_id: ID of the user
        :param k: Number of recommendations
        :return: List of recommended movie titles
        """
        user_ratings = self.dataset_ratings[self.dataset_ratings['userId'] == user_id]
        
        if user_ratings.empty:
            print(f"No ratings found for user {user_id}")
            return []
        
        # Find the movie with the highest rating
        top_movie_id = user_ratings.loc[user_ratings['rating'].idxmax(), 'movieId']
        
        # Find similar movies
        similar_movie_ids = self.find_similar_movies(top_movie_id, k)
        
        # Get movie titles
        recommendations = [self.movie_titles.get(movie_id, "Unknown Movie") for movie_id in similar_movie_ids]
        
        return recommendations

# Example usage
if __name__ == "__main__":
    recommender = MovieRecommender('data/movies.csv', 'data/ratings.csv')
    
    # Get recommendations for a user
    user_id = 1
    recommendations = recommender.recommend_for_user(user_id)
    
    print(f"Recommendations for User {user_id}:")
    for movie in recommendations:
        print(movie)
