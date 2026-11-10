import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class LLMRecSys:
    def __init__(self, movies_df):
        self.movies = movies_df.head(1500)
        self.descriptions = self.movies['overview'].tolist()
        self.titles = self.movies['title'].tolist()

        print("Загрузка эмбеддинг-модели")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.embeddings = self.model.encode(self.descriptions, show_progress_bar = False)

    def recommend(self, user_history_descriptions, n=3):
        print("Retrieval: поиск кандидатов по эмбеддингам")
        history_emb = self.model.encode(user_history_descriptions)
        user_profile = np.mean(history_emb, axis=0).reshape(1, -1)

        sims = cosine_similarity(user_profile, self.embeddings)[0]
        top_5_idx = sims.argsort()[-5:][::-1]

        candidates = [(self.titles[i], self.descriptions[i]) for i in top_5_idx]

        print("Generation: отправка в LLM")
        prompt = f"""История пользователя:
{chr(10).join(f'- {h}' for h in user_history_descriptions)}

Кандидаты:
{chr(10).join(f'{i + 1}. {c[0]} (Описание: {c[1]})' for i, c in enumerate(candidates))}

Выбери {n} лучших фильма и напиши обоснование (1 предложение на каждый).
Отвечай на русском!
"""
        try:
            res = requests.post("http://localhost:11434/api/generate",
                                json={"model": "gemma2:2b", "prompt": prompt, "stream": False},
                                timeout=120)
            return res.json().get('response', 'Ошибка LLM')
        except Exception as e:
            return f"Ошибка соединения с Ollama: {e}"