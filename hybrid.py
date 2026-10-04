import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
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
movies = movies.drop_duplicates(subset = 'id')

id_to_title = dict(zip(movies['id'], movies['title']))
title_to_id = dict(zip(movies['title'], movies['id']))
print(f"{len(merged)} оценок, {len(movies)} фильмов")

def parse_genres(g):
    try:
        return [x['name'] for x in ast.literal_eval(g)]
    except:
        return []

movies['genres_list'] = movies['genres'].apply(parse_genres)
movies['genres_str'] = movies['genres_list'].apply(lambda x: ' '.join(x))

rated_ids = set(merged['tmdbId'].unique())
movies_filtered = movies[movies['id'].isin(rated_ids)].reset_index(drop = True)
print(f"{len(movies_filtered)} фильмов с оценками и жанрами")
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), token_pattern = None)
genre_matrix = vectorizer.fit_transform(movies_filtered['genres_str'])
content_ids = movies_filtered['id'].tolist()
content_id_to_row = {mid: i for i, mid in enumerate(content_ids)}

movie_user_matrix = merged.pivot_table(
    index = 'tmdbId',
    columns = 'userId',
    values = 'rating'
).fillna(0)

collab_ids = movie_user_matrix.index.tolist()
collab_id_to_row = {mid: i for i, mid in enumerate(collab_ids)}

common_ids = list(set(content_ids) & set(collab_ids))
print(f"Кол-во общих фильмов - {len(common_ids)}")


def recommend_hybrid(title, n=10, weight_content = 0.5):
    if title not in title_to_id:
        print(f"Фильм {title} не найден")
        return []

    tmdb_id = title_to_id[title]
    if tmdb_id not in content_id_to_row or tmdb_id not in collab_id_to_row:
        print(f"Для фильма {title} нет данных в одном из подходов")
        return []

    c_idx = content_id_to_row[tmdb_id]
    content_sim = cosine_similarity(
        genre_matrix[c_idx:c_idx + 1],
        genre_matrix
    ).flatten()

    col_idx = collab_id_to_row[tmdb_id]
    collab_sim = cosine_similarity(
        movie_user_matrix.iloc[col_idx:col_idx + 1],
        movie_user_matrix
    ).flatten()

    scores = {}
    for i, mid in enumerate(content_ids):
        scores[mid] = weight_content * content_sim[i]
    for i, mid in enumerate(collab_ids):
        if mid in scores:
            scores[mid] += (1 - weight_content) * collab_sim[i]
        else:
            scores[mid] = (1 - weight_content) * collab_sim[i]
    scores.pop(tmdb_id, None)
    top = sorted(scores.items(), key = lambda x: x[1], reverse = True)[:n]
    result = []
    for mid, score in top:
        name = id_to_title.get(mid, f"ID {mid}")
        result.append((name, round(score, 3)))
    return result

if __name__ == "__main__":
    test_titles = ["The Holiday", "The Lion King", "The Godfather"]
    for title in test_titles:
        print(f"\n{'-'*60}")
        print(f"Гибридные рекомендации для {title}")
        print('-'*60)
        recs = recommend_hybrid(title, n = 10, weight_content = 0.5)
        if recs:
            for i, (name, score) in enumerate(recs, 1):
                print(f"{i}. {name}  (score: {score})")
        else:
            print("Ничего не найдено")