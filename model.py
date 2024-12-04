import numpy as np
from keras.models import load_model

# Загрузка модели
model = load_model('iris_model.h5')

def predict_iris(features):
    try:
        # Преобразуем список в numpy массив
        features_array = np.array(features)
        # Убедимся, что данные имеют нужную форму
        features_array = features_array.reshape(1, -1)
        prediction = model.predict(features_array)
        
        # Преобразуем результат в тип, который можно сериализовать (например, int)
        predicted_class = prediction.argmax(axis=1)[0]  # Индекс класса с максимальной вероятностью
        
        return {"species": int(predicted_class)}  # Возвращаем как целое число
    except Exception as e:
        return {"error": f"Error in prediction: {e}"}
