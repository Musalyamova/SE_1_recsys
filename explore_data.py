import pandas as pd

movies = pd.read_csv('data/movies_metadata.csv', low_memory = False)
ratings = pd.read_csv('data/ratings_small.csv')
links = pd.read_csv('data/links_small.csv')

print("movies_metadata")
print("Размер")
print(movies.shape)
print("Колонки")
print(list(movies.columns))
print()

print("ratings_small")
print("Размер")
print(ratings.shape)
print("Колонки")
print(list(ratings.columns))

print("links_small")
print("Размер")
print(links.shape)
print("Колонки")
print(list(links.columns))

print()
print(movies[['id', 'title', 'genres']].head(3))
print()
print(ratings.head(3))