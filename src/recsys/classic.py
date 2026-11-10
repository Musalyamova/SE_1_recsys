import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecSys:
    def __init__(self, movies_df):
        self.movies = movies_df.copy()
        self._prepare_data()

    def _prepare_data(self):
        def parse_genres(g):
            try:
                return ' '.join([x['name'] for x in ast.literal_eval(g)])
            except:
                return ''

        self.movies['genres_str'] = self.movies['genres'].apply(parse_genres)

        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), token_pattern=None)
        self.genre_matrix = vectorizer.fit_transform(self.movies['genres_str'])
        self.movies = self.movies[self.genre_matrix.getnnz(axis=1) > 0]
        self.genre_matrix = self.genre_matrix[self.genre_matrix.getnnz(axis=1) > 0]
        self.indices = self.movies.reset_index().set_index('title')['index']

    def recommend(self, title, n=5):
        if title not in self.indices:
            return []
        idx = self.indices[title]
        sim_scores = cosine_similarity(self.genre_matrix[idx], self.genre_matrix).flatten()
        top_indices = sim_scores.argsort()[-n - 1:-1][::-1]
        return self.movies.iloc[top_indices]['title'].tolist()