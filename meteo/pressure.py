import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Пример данных: атмосферное давление (в гПа) за последние 30 дней
pressure_data = [1012, 1013, 1011, 1010, 1014, 1016, 1015, 
                 1017, 1019, 1020, 1018, 1017, 1015, 1013, 
                 1014, 1016, 1017, 1018, 1019, 1021, 1022, 
                 1021, 1019, 1018, 1017, 1015, 1014, 1012, 
                 1011, 1010]  # Примерные данные

# Размер окна для скользящего прогноза (7 дней)
window_size = 7

# Подготовка данных для тренировки модели
X = []
y = []

# Формируем обучающие и целевые данные с использованием скользящего окна
for i in range(len(pressure_data) - window_size):
    X.append(pressure_data[i:i + window_size])  # Давление за предыдущие 7 дней
    y.append(pressure_data[i + window_size])    # Давление на следующий день

X = np.array(X)
y = np.array(y)

# Создаем и обучаем модель линейной регрессии
model = LinearRegression()
model.fit(X, y)

# Прогнозируем давление на следующий день
last_window = np.array(pressure_data[-window_size:]).reshape(1, -1)
predicted_pressure = model.predict(last_window)

print(f"Прогнозируемое давление на следующий день: {predicted_pressure[0]:.2f} гПа")

# Оценка точности модели
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Среднеквадратическая ошибка модели: {mse:.2f}")

# Графическое отображение данных
plt.figure(figsize=(10, 10))

# Исторические данные атмосферного давления
plt.plot(range(len(pressure_data)), pressure_data, label="Историческое давление", marker='o', color='b')
# Прогнозируемое давление
plt.plot(len(pressure_data), predicted_pressure[0], 'ro', label="Прогнозируемое давление")
plt.axvline(len(pressure_data) - 1, color='gray', linestyle='--')  # линия, разделяющая реальные данные и прогноз

# Оформление графика
plt.title("Прогноз давления на основе исторических данных")
plt.xlabel("Дни")
plt.ylabel("Атмосферное давление (гПа)")
plt.legend()
plt.grid()
plt.show()
