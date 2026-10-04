import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('data/ratings_small.csv')
links = pd.read_csv('data/links_small.csv')
movies = pd.read_csv('data/movies_metadata.csv', low_memory = False)

merged = ratings.merge(links, on = 'movieId', how = 'inner')
merged['tmdbId'] = pd.to_numeric(merged['tmdbId'], errors = 'coerce')
merged = merged.dropna(subset = ['tmdbId'])
merged['tmdbId'] = merged['tmdbId'].astype(int)
movies['id'] = pd.to_numeric(movies['id'], errors = 'coerce')
movies = movies.dropna(subset = ['id'])
movies['id'] = movies['id'].astype(int)

id_to_title = dict(zip(movies['id'], movies['title']))
title_to_id = dict(zip(movies['title'], movies['id']))

print(f"Кол-во оценок: {len(merged)}")
print(f"Кол-во пользователей: {merged['userId'].nunique()}")
print(f"Кол-во фильмов: {merged['tmdbId'].nunique()}")

movie_user_matrix = merged.pivot_table(
    index='tmdbId',
    columns='userId',
    values='rating'
).fillna(0)

print(f"Матрица: {movie_user_matrix.shape}")

movie_ids = movie_user_matrix.index.tolist()
id_to_row = {mid: i for i, mid in enumerate(movie_ids)}


def recommend_by_movie(title, n=10):
    """Item-based - рекомендует фильмы, которые похожи на указанный по оценкам других пользователей"""
    if title not in title_to_id:
        print(f"Фильм '{title}' не найден в базе")
        return []

    tmdb_id = title_to_id[title]
    if tmdb_id not in id_to_row:
        print(f"Для фильма '{title}' нет оценок в датасете")
        return []

    row_idx = id_to_row[tmdb_id]
    sim_scores = cosine_similarity(
        movie_user_matrix.iloc[row_idx:row_idx+1],
        movie_user_matrix
    ).flatten()

    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:n+1]

    result = []
    for idx, score in sim_scores:
        mid = movie_ids[idx]
        name = id_to_title.get(mid, f"ID {mid}")
        result.append((name, round(score, 3)))
    return result

if __name__ == "__main__":
    test_titles = ["Interstellar", "Fight Club", "Desperate Housewives"]
    for title in test_titles:
        print(f"\n{'-'*60}")
        print(f"На фильм {title} похожи:")
        print('-'*60)
        recs = recommend_by_movie(title, n = 10)
        if recs:
            for i, (name, score) in enumerate(recs, 1):
                print(f"{i}. {name}  (близость: {score})")
        else:
            print("Ничего не найдено")