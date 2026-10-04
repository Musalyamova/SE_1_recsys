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
print(f"Всего оценок: {len(merged)}")

np.random.seed(42)
msk = np.random.rand(len(merged)) < 0.8
train = merged[msk]
test = merged[~msk]
print(f"Train {len(train)}, Test {len(test)}")

movie_user_matrix = train.pivot_table(
    index='tmdbId',
    columns='userId',
    values='rating'
).fillna(0)
movie_ids = movie_user_matrix.index.tolist()
id_to_row = {mid: i for i, mid in enumerate(movie_ids)}
print(f"Матрица train: {movie_user_matrix.shape}")

def recommend_for_user(user_id, n=10):
    if user_id not in movie_user_matrix.columns:
        return []
    user_col = movie_user_matrix[user_id]
    seen = user_col[user_col > 0].index.tolist()
    if not seen:
        return []
    scores = np.zeros(len(movie_ids))
    for mid in seen:
        row_idx = id_to_row[mid]
        rating = user_col[mid]
        sim = cosine_similarity(
            movie_user_matrix.iloc[row_idx: row_idx + 1],
            movie_user_matrix
        ).flatten()
        scores += sim * rating
    for mid in seen:
        scores[id_to_row[mid]] = -1
    top_idx = np.argsort(scores)[::-1][:n]
    return [movie_ids[i] for i in top_idx]

def precision_recall_at_k(k = 10, n_users = 100):
    precisions = []
    recalls = []
    test_users = test['userId'].unique()[:n_users]
    for user_id in test_users:
        user_test = test[(test['userId'] == user_id) & (test['rating'] >= 4.0)]
        relevant = set(user_test['tmdbId'].tolist())
        if not relevant:
            continue
        recs = recommend_for_user(user_id, n = k)
        if not recs:
            continue
        hits = len(set(recs) & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)

    print(f"\nИз {len(test_users)} учтено {len(precisions)} пользователей")
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}:    {avg_recall:.4f}")
    return avg_precision, avg_recall

if __name__ == "__main__":
    for k in [5, 10, 20]:
        print(f"\n{'-'*60}")
        print(f"Метрики для K = {k}")
        print('-'*60)
        precision_recall_at_k(k = k, n_users = 100)