import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Пример данных: температура океана (в °C), давление (в гПа) и интенсивность циклона (скорость ветра в км/ч) за последние 30 дней
ocean_temp_data = [28.2, 28.5, 28.7, 28.9, 29.1, 29.2, 29.3, 
                   29.4, 29.5, 29.6, 29.7, 29.8, 29.9, 30.0,
                   30.1, 30.2, 30.3, 30.4, 30.5, 30.4, 30.3, 
                   30.2, 30.1, 30.0, 29.9, 29.8, 29.7, 29.6,
                   29.5, 29.4]
pressure_data = [1008, 1007, 1006, 1005, 1004, 1003, 1002, 
                 1001, 1000, 999, 998, 997, 996, 995, 
                 994, 993, 992, 991, 990, 991, 992, 
                 993, 994, 995, 996, 997, 998, 999, 
                 1000, 1001]
intensity_data = [80, 85, 87, 89, 92, 95, 98, 
                  100, 102, 104, 106, 108, 110, 112, 
                  114, 115, 116, 117, 118, 116, 114, 
                  112, 110, 108, 106, 104, 102, 100, 
                  98, 96]  # Скорость ветра в центре циклона в км/ч

# Размер окна для скользящего прогноза (7 дней)
window_size = 7

# Подготовка данных для тренировки модели
X = []
y = []

# Создаем обучающие данные, используя скользящее окно на 7 дней
for i in range(len(ocean_temp_data) - window_size):
    # Входные данные — температура океана и давление за последние 7 дней
    X.append(ocean_temp_data[i:i + window_size] + pressure_data[i:i + window_size])
    # Целевое значение — интенсивность циклона на следующий день
    y.append(intensity_data[i + window_size])

X = np.array(X)
y = np.array(y)

# Создаем и обучаем модель линейной регрессии
model = LinearRegression()
model.fit(X, y)

# Прогнозируем интенсивность циклона на следующий день
last_window = np.array(ocean_temp_data[-window_size:] + pressure_data[-window_size:]).reshape(1, -1)
predicted_intensity = model.predict(last_window)

print(f"Прогнозируемая интенсивность циклона на следующий день: {predicted_intensity[0]:.2f} км/ч")

# Оценка точности модели
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Среднеквадратическая ошибка модели: {mse:.2f}")

# Графическое отображение данных
plt.figure(figsize=(10, 10))

# Исторические данные интенсивности циклона
plt.plot(range(len(intensity_data)), intensity_data, label="Историческая интенсивность", marker='o', color='b')
# Прогнозируемая интенсивность на следующий день
plt.plot(len(intensity_data), predicted_intensity[0], 'ro', label="Прогнозируемая интенсивность")
plt.axvline(len(intensity_data) - 1, color='gray', linestyle='--')  # линия, разделяющая реальные данные и прогноз

# Оформление графика
plt.title("Прогноз интенсивности циклона на основе исторических данных")
plt.xlabel("Дни")
plt.ylabel("Интенсивность циклона (км/ч)")
plt.legend()
plt.grid()
plt.show()
