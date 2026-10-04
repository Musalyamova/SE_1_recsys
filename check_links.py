import pandas as pd

ratings = pd.read_csv('data/ratings_small.csv')
links = pd.read_csv('data/links_small.csv')
movies = pd.read_csv('data/movies_metadata.csv', low_memory = False)

print(ratings.shape)
print(links.shape)
print(movies.shape)

merged = ratings.merge(links, on = 'movieId', how = 'inner')
print(f"После {merged.shape}")
print(merged[['userId', 'movieId', 'rating', 'tmdbId']].head())
merged['tmdbId'] = pd.to_numeric(merged['tmdbId'], errors = 'coerce')
print("\nТипы", merged['tmdbId'].dtype)
print(merged['tmdbId'].isna().sum())
print(merged['userId'].nunique())
print(merged['tmdbId'].nunique())