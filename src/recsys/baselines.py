class PopularityBaseline:
    def __init__(self, ratings_df):
        self.ratings_df = ratings_df

    def recommend(self, n=5):
        stats = self.ratings_df.groupby('tmdbId').agg(
            count=('rating', 'count'),
            avg=('rating', 'mean')
        )
        stats = stats[stats['avg'] >= 3.5]
        top = stats.sort_values('count', ascending=False).head(n)
        return top.index.tolist()