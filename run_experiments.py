from src.recsys.data import DataLoader
from src.recsys.baselines import PopularityBaseline
from src.recsys.classic import ContentBasedRecSys
from src.recsys.llm_recsys import LLMRecSys

def main():
    print("--- 1. Загрузка данных ---")
    loader = DataLoader()
    movies = loader.load_movies()
    ratings = loader.load_ratings()
    print(f"Загружено {len(movies)} фильмов и {len(ratings)} оценок.\n")

    print("--- 2. Baseline (Популярное) ---")
    baseline = PopularityBaseline(ratings)
    print("Топ-5 ID популярных фильмов:", baseline.recommend(5), "\n")

    print("--- 3. Классический Content-based ---")
    cb = ContentBasedRecSys(movies)
    print("Похожие на 'Interstellar':", cb.recommend("Interstellar", 3), "\n")

    print("--- 4. Продвинутый LLM / RAG подход ---")
    history = [
        "A thief who steals corporate secrets through the use of dream-sharing technology.",
        "Batman must accept one of the greatest psychological and physical tests."
    ]
    llm = LLMRecSys(movies)
    result = llm.recommend(history, n=3)
    print("\nРезультат ОТ LLM:\n", result)

if __name__ == "__main__":
    main()