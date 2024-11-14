import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Пример данных: температура (°C), влажность (%) и скорость ветра (м/с) и уровень загрязнения (PM2.5)
# Замените эти данные на реальные, если они у вас есть
data = {
    'Temperature': [15, 17, 19, 21, 23, 25, 27, 29, 31, 33],  # Температура воздуха (°C)
    'Humidity': [45, 47, 50, 53, 56, 58, 60, 62, 65, 67],  # Влажность (%)
    'WindSpeed': [3.0, 2.8, 3.2, 3.5, 3.8, 4.0, 4.2, 4.5, 4.8, 5.0],  # Скорость ветра (м/с)
    'PM25': [35, 40, 45, 50, 55, 60, 70, 75, 80, 85]  # Уровень загрязнения (PM2.5, мкг/м³)
}

# Создание DataFrame из данных
df = pd.DataFrame(data)

# Определяем независимые переменные (температура, влажность, скорость ветра) и зависимую переменную (уровень загрязнения)
X = df[['Temperature', 'Humidity', 'WindSpeed']]  # независимые переменные (температура, влажность, скорость ветра)
y = df['PM25']  # зависимая переменная (уровень загрязнения)

# Создаем модель линейной регрессии
model = LinearRegression()

# Обучаем модель на данных
model.fit(X, y)

# Получаем коэффициенты линейной регрессии
intercept = model.intercept_  # свободный член (β0)
coefficients = model.coef_  # коэффициенты наклона (β1, β2, β3)

# Выводим уравнение линейной регрессии
print(f"Уравнение линейной регрессии: PM25 = {intercept:.2f} + "
      f"{coefficients[0]:.2f} * Temperature + "
      f"{coefficients[1]:.2f} * Humidity + "
      f"{coefficients[2]:.2f} * WindSpeed")

# Прогнозируем уровень загрязнения на основе значений независимых переменных
y_pred = model.predict(X)

# Визуализируем результаты
fig = plt.figure(figsize=(10, 10))

# График для отображения зависимости PM2.5 от температуры
plt.subplot(1, 2, 1)
plt.scatter(df['Temperature'], df['PM25'], color='blue', label='Данные')
plt.plot(df['Temperature'], y_pred, color='red', label='Прогноз')
plt.title('Загрязнение воздуха в зависимости от температуры')
plt.xlabel('Температура (°C)')
plt.ylabel('PM2.5 (мкг/м³)')
plt.legend()

# График для отображения зависимости PM2.5 от влажности
plt.subplot(1, 2, 2)
plt.scatter(df['Humidity'], df['PM25'], color='blue', label='Данные')
plt.plot(df['Humidity'], y_pred, color='red', label='Прогноз')
plt.title('Загрязнение воздуха в зависимости от влажности')
plt.xlabel('Влажность (%)')
plt.ylabel('PM2.5 (мкг/м³)')
plt.legend()

plt.tight_layout()
plt.show()
