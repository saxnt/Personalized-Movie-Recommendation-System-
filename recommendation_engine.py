import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

class MovieRecommender:
    def __init__(self, movies_path, ratings_path):
        self.dataset_movies = pd.read_csv(movies_path)
        self.dataset_ratings = pd.read_csv(ratings_path)
        self.movie_titles = dict(zip(self.dataset_movies['movieId'], self.dataset_movies['title']))
        
        self.X, self.user_map_to_index, self.movie_map_to_index, self.user_inv, self.movie_inv = self.create_sparse_matrix()

    def create_sparse_matrix(self):
        n = len(self.dataset_ratings['userId'].unique())
        m = len(self.dataset_ratings['movieId'].unique())
        
        user_map_to_index = dict(zip(np.unique(self.dataset_ratings['userId']), list(range(n))))
        movie_map_to_index = dict(zip(np.unique(self.dataset_ratings['movieId']), list(range(m))))
        
        user_inv = dict(zip(list(range(n)), self.dataset_ratings['userId'].unique()))
        movie_inv = dict(zip(list(range(m)), self.dataset_ratings['movieId'].unique()))
        
        user_index = [user_map_to_index[i] for i in self.dataset_ratings['userId']]
        movie_index = [movie_map_to_index[i] for i in self.dataset_ratings['movieId']]
        
        X = csr_matrix((self.dataset_ratings['rating'], (movie_index, user_index)), shape=(m, n))
        
        return X, user_map_to_index, movie_map_to_index, user_inv, movie_inv

    def find_similar_movies(self, movie_id, k=10, metric='cosine'):
        neighbor_id = []
        movie_ind = self.movie_map_to_index[movie_id]
        movie_vec = self.X[movie_ind]
        k = k + 1
        
        knn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric=metric)
        knn.fit(self.X)
        movie_vec = movie_vec.reshape(1, -1)
        
        neighbour = knn.kneighbors(movie_vec, return_distance=False)
        
        for i in range(1, k):
            n = neighbour[0][i]
            neighbor_id.append(self.movie_inv[n])
        
        return neighbor_id

    def recommend_movies(self, user_id, k=10):
        df1 = self.dataset_ratings[self.dataset_ratings['userId'] == user_id]
        
        if df1.empty:
            print(f"User with ID {user_id} does not exist")
            return
        
        movie_id = df1[df1['rating'] == max(df1['rating'])]['movieId'].iloc[0]
        
        similar_ids = self.find_similar_movies(movie_id, k)
        movie_title = self.movie_titles.get(movie_id, "Movie not found")
        
        if movie_title == "Movie not found":
            print(f"Movie with ID {movie_id} not found")
            return
        
        print(f"Since you watched {movie_title}, you might also like: ")
        for movie_id in similar_ids:
            print(self.movie_titles.get(movie_id, "Movie not found"))
