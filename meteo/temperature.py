import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Пример данных: среднесуточная температура за последние 30 дней
temperature_data = [15.5, 16.1, 15.8, 15.2, 14.8, 15.0, 15.3, 
                    15.6, 15.9, 16.3, 16.5, 17.0, 17.2, 17.5,
                    17.8, 18.0, 18.2, 18.5, 18.7, 19.0, 19.1,
                    19.4, 19.5, 19.6, 19.5, 19.3, 19.1, 18.9,
                    18.7, 18.5]  # Примерные данные

# Размер окна для скользящего прогноза (7 дней)
window_size = 7

# Подготовка данных для тренировки модели
X = []
y = []

# Формируем обучающие и целевые данные с использованием скользящего окна
for i in range(len(temperature_data) - window_size):
    X.append(temperature_data[i:i + window_size])  # Предыдущие 7 дней температуры
    y.append(temperature_data[i + window_size])    # Температура на следующий день

X = np.array(X)
y = np.array(y)

# Создаем и обучаем модель линейной регрессии
model = LinearRegression()
model.fit(X, y)

# Прогнозируем температуру на следующий день
last_window = np.array(temperature_data[-window_size:]).reshape(1, -1)
predicted_temp = model.predict(last_window)

print(f"Прогнозируемая температура на следующий день: {predicted_temp[0]:.2f} °C")

# Оценка точности модели
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Среднеквадратическая ошибка модели: {mse:.2f}")

# Графическое отображение данных
plt.figure(figsize=(10, 10))

# Исторические данные температуры
plt.plot(range(len(temperature_data)), temperature_data, label="Историческая температура", marker='o', color='b')
# Прогнозируемая температура
plt.plot(len(temperature_data), predicted_temp[0], 'ro', label="Прогнозируемая температура")
plt.axvline(len(temperature_data) - 1, color='gray', linestyle='--')  # линия, разделяющая реальные данные и прогноз

# Оформление графика
plt.title("Прогноз температуры на основе исторических данных")
plt.xlabel("Дни")
plt.ylabel("Температура (°C)")
plt.legend()
plt.grid()
plt.show()
