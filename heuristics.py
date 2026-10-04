import pandas as pd
from collections import Counter

ratings = pd.read_csv('data/ratings_small.csv')
links = pd.read_csv('data/links_small.csv')
movies = pd.read_csv('data/movies_metadata.csv', low_memory = False)

merged = ratings.merge(links, on = "movieId", how = "inner")
merged['tmdbId'] = pd.to_numeric(merged['tmdbId'], errors = 'coerce')
merged = merged.dropna(subset = ['tmdbId'])
merged['tmdbId'] = merged['tmdbId'].astype(int)
movies['id'] = pd.to_numeric(movies['id'], errors = 'coerce')
movies = movies.dropna(subset = ['id'])
movies['id'] = movies['id'].astype(int)
id_to_title = dict(zip(movies['id'], movies['title']))
title_to_id = dict(zip(movies['title'], movies['id']))
print(f"Всего оценок: {len(merged)}")

def popular_movies(n = 10, min_rating = 3.5):
    stats = merged.groupby('tmdbId').agg(
        count = ('rating', 'count'),
        avg = ('rating', 'mean')
    )
    stats = stats[stats['avg'] >= min_rating]
    stats = stats.sort_values('count', ascending = False).head(n)
    result = []
    for mid, row in stats.iterrows():
        mid = int(mid)
        name = id_to_title.get(mid, f"id {mid}")
        result.append((name, int(row['count']), round(row['avg'], 2)))
    return result

def also_watched(title, n = 10):
    if title not in title_to_id:
        print("Фильм не найден")
        return []
    target_id = title_to_id[title]
    target_users = set(merged[merged['tmdbId'] == target_id]['userId'])

    if not target_users:
        print(f"Этот фильм {title} никто не смотрел")
        return []

    other = merged[merged['userId'].isin(target_users) & (merged['tmdbId'] != target_id)]
    counts = Counter(other['tmdbId'])
    top = counts.most_common(n)
    result = []
    for mid, cnt in top:
        mid = int(mid)
        name = id_to_title.get(mid, f"id {mid}")
        result.append((name, cnt))
    return result

if __name__ == "__main__":
    print("\n" + "-"*60)
    print("Популярные фильмы:")
    pop = popular_movies(n = 10)
    for i, (name, cnt, avg) in enumerate(pop, 1):
        print(f"{i}. {name} (средняя оценка: {avg}, всего оценок: {cnt})")
    print("\n" + "-" * 60)
    print("С этим фильмом также смотрят:")
    print("-"*60)
    for title in ["The Matrix", "Avatar", "Titanic"]:
        print(f"\n С {title} вместе смотрели:")
        recs = also_watched(title, n = 5)
        for i, (name, cnt) in enumerate(recs, 1):
            print(f"{i}. {name}, общих пользователей {cnt}")
