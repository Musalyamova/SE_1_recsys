# RecSys
Система рекомендаций для фильмов на стриминге. Реализованные подходы:
- content-based
- collaborative
- гибридный
- эвристики
- метрики

## Стек
- Python 3.13
- pandas, numpy, scikit-learn

## Датасет
The Movies Dataset (TMDB + MovieLens),  всего 45 000 фильмов и 100 000 оценок
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
Файлы (`movies_metadata.csv`, `ratings_small.csv`, `links_small.csv`) положить в `data/`

## Подходы
| Подход                       | Файл                  |
|------------------------------|-----------------------|
| Content-based                | content_based.py      |
| Коллаборативный (item-based) | collaborative.py      |
| Коллаборативный (user-based) | collaborative_user.py |
| Гибридный                    | hybrid.py             |
| Эвристики                    | heuristics.py         |
| Метрики                      | metrics.py            |
| Анализ данных                | explore_data.py       |
| Проверка связей              | check_links.py        |
| Анализ проблем               | analysis.md           |

## Запуск
```bash
pip install -r requirements.txt
python hybrid.py
```