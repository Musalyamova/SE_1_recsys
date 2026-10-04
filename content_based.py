import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('data/movies_metadata.csv', low_memory=False)
print(f"До: {len(movies)} фильмов")

movies = movies[['id', 'title', 'genres']].dropna()
movies = movies.drop_duplicates(subset='id')
print(f"После: {len(movies)} фильмов")

def parse_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g['name'] for g in genres]
    except:
        return []

movies['genres_list'] = movies['genres'].apply(parse_genres)
movies['genres_str'] = movies['genres_list'].apply(lambda x: ' '.join(x))

vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), token_pattern = None)
genre_matrix = vectorizer.fit_transform(movies['genres_str'])
print(genre_matrix.shape)

movies = movies[genre_matrix.getnnz(axis = 1) > 0].reset_index(drop = True)
genre_matrix = genre_matrix[genre_matrix.getnnz(axis = 1) > 0]
print(f"Без жанров:{len(movies)} фильмов")
indices = pd.Series(movies.index, index = movies['title']).drop_duplicates()

def recommend_by_genre(title, n = 10):
    if title not in indices:
        print(f"{title} не найден в базе")
        return []
    idx = indices[title]
    sim_scores = cosine_similarity(genre_matrix[idx], genre_matrix).flatten()
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

if __name__ == "__main__":
    test_titles = ["Interstellar", "Fight Club", "Bridgertons"]
    for title in test_titles:
        print('-' * 60)
        print(f"Похожие на {title}:")
        print('-'*60)
        recs = recommend_by_genre(title, n = 10)
        if recs:
            for i, rec in enumerate(recs, 1):
                print(f"{i}. {rec}")
        else:
            print("Ничего не найдено")
