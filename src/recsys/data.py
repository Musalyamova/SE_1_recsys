import pandas as pd
import os


class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def load_movies(self):
        path = os.path.join(self.data_dir, 'movies_metadata.csv')
        df = pd.read_csv(path, low_memory=False)
        df['id'] = pd.to_numeric(df['id'], errors='coerce').dropna()
        return df.dropna(subset=['id', 'title', 'overview'])

    def load_ratings(self):
        ratings = pd.read_csv(os.path.join(self.data_dir, 'ratings_small.csv'))
        links = pd.read_csv(os.path.join(self.data_dir, 'links_small.csv'))

        merged = ratings.merge(links, on='movieId', how='inner')
        merged['tmdbId'] = pd.to_numeric(merged['tmdbId'], errors='coerce')
        return merged.dropna(subset=['tmdbId'])