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
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies = movies.dropna(subset=['id'])
movies['id'] = movies['id'].astype(int)
id_to_title = dict(zip(movies['id'], movies['title']))

print(f"Кол-во оценок: {len(merged)}")
print(f"Кол-во пользователей: {merged['userId'].nunique()}")
print(f"Кол-во фильмов: {merged['tmdbId'].nunique()}")

user_movie_matrix = merged.pivot_table(
    index='userId',
    columns='tmdbId',
    values='rating'
).fillna(0)

print(f"Матрица: {user_movie_matrix.shape}")
user_ids = user_movie_matrix.index.tolist()
movie_ids = user_movie_matrix.columns.tolist()
user_to_row = {uid: i for i, uid in enumerate(user_ids)}


def recommend_for_user(user_id, n=10):
    """User-based - предлагает фильмы на основе похожих фильмов у других пользователей"""
    if user_id not in user_to_row:
        print(f"Пользователь {user_id} не найден.")
        return []
    row_idx = user_to_row[user_id]
    sim_scores = cosine_similarity(
        user_movie_matrix.iloc[row_idx:row_idx+1],
        user_movie_matrix
    ).flatten()
    similar_users = np.argsort(sim_scores)[::-1][1:21]
    our_ratings = user_movie_matrix.iloc[row_idx]
    similar_ratings = user_movie_matrix.iloc[similar_users]
    mean_ratings = similar_ratings.mean(axis=0)
    unseen = our_ratings[our_ratings == 0].index
    mean_ratings = mean_ratings[unseen]
    top_movies = mean_ratings.sort_values(ascending=False).head(n)
    result = []
    for mid, score in top_movies.items():
        name = id_to_title.get(mid, f"ID {mid}")
        result.append((name, round(score, 3)))
    return result

if __name__ == "__main__":
    test_users = [1, 100, 500]
    for user_id in test_users:
        print(f"\n{'-'*60}")
        print(f"Рекомендации для пользователя {user_id}:")
        print('-'*60)
        user_ratings = merged[merged['userId'] == user_id]
        print("Уже смотрел:")
        top_seen = user_ratings.sort_values('rating', ascending = False).head(5)
        for i, row in top_seen.iterrows():
            title = id_to_title.get(row['tmdbId'], f"ID {row['tmdbId']}")
            print(f"- {title} ({row['rating']})")
        print("\nРекомендуем")
        recs = recommend_for_user(user_id, n = 10)
        if recs:
            for i, (name, score) in enumerate(recs, 1):
                print(f"{i}. {name}  (средняя оценка похожих фильмов: {score})")
        else:
            print("Ничего не найдено")