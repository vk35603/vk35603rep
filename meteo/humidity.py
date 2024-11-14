import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Пример данных: относительная влажность (в %) за последние 30 дней
humidity_data = [70, 72, 68, 65, 67, 69, 70, 
                 71, 73, 75, 74, 76, 78, 77, 
                 76, 75, 74, 73, 72, 71, 70,
                 72, 73, 74, 76, 75, 74, 73,
                 72, 71]  # Примерные данные

# Размер окна для скользящего прогноза (7 дней)
window_size = 7

# Подготовка данных для тренировки модели
X = []
y = []

# Формируем обучающие и целевые данные с использованием скользящего окна
for i in range(len(humidity_data) - window_size):
    X.append(humidity_data[i:i + window_size])  # Предыдущие 7 дней влажности
    y.append(humidity_data[i + window_size])    # Влажность на следующий день

X = np.array(X)
y = np.array(y)

# Создаем и обучаем модель линейной регрессии
model = LinearRegression()
model.fit(X, y)

# Прогнозируем влажность на следующий день
last_window = np.array(humidity_data[-window_size:]).reshape(1, -1)
predicted_humidity = model.predict(last_window)

print(f"Прогнозируемая влажность на следующий день: {predicted_humidity[0]:.2f}%")

# Оценка точности модели
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Среднеквадратическая ошибка модели: {mse:.2f}")

# Графическое отображение данных
plt.figure(figsize=(10, 10))
plt.plot(range(len(humidity_data)), humidity_data, label="Историческая влажность", marker='o')
plt.plot(len(humidity_data), predicted_humidity[0], 'ro', label="Прогнозируемая влажность")
plt.axvline(len(humidity_data) - 1, color='gray', linestyle='--')  # линия, разделяющая реальные данные и прогноз

# Оформление графика
plt.title("Прогноз влажности на основе исторических данных")
plt.xlabel("Дни")
plt.ylabel("Влажность (%)")
plt.legend()
plt.grid()
plt.show()
