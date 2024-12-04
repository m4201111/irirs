# main.py

from fastapi import FastAPI
from app.model import predict_iris  # Импортируем функцию из model.py

# Создаём экземпляр приложения FastAPI
app = FastAPI()

# Определяем маршрут для главной страницы
@app.get("/")
def get_iris_prediction(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    """
    Принимает параметры, связанные с цветком ириса, и возвращает предсказание вида.
    :param sepal_length: Длина чашелистика
    :param sepal_width: Ширина чашелистика
    :param petal_length: Длина лепестка
    :param petal_width: Ширина лепестка
    :return: Словарь с предсказанным видом ириса
    """
    # Вызов функции для предсказания вида ириса
    result = predict_iris([sepal_length, sepal_width, petal_length, petal_width])
    return {"species": result}  # Возвращаем результат в формате JSON
