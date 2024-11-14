import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Пример данных: концентрация CO2 в атмосфере (в ppm) и средняя температура на Земле (°C)
# Замените эти данные на реальные, если они у вас есть
data = {
    'CO2': [280, 290, 300, 310, 320, 330, 340, 350, 360, 370],  # Концентрация CO2 в ppm
    'Temperature': [14.0, 14.1, 14.2, 14.3, 14.5, 14.7, 14.9, 15.0, 15.2, 15.4]  # Средняя температура на Земле (°C)
}

# Создание DataFrame из данных
df = pd.DataFrame(data)

# Определяем независимую переменную (концентрация CO2) и зависимую переменную (температура)
X = df[['CO2']]  # независимая переменная (CO2)
y = df['Temperature']  # зависимая переменная (температура)

# Создаем модель линейной регрессии
model = LinearRegression()

# Обучаем модель на данных
model.fit(X, y)

# Получаем коэффициенты линейной регрессии
intercept = model.intercept_  # свободный член (β0)
slope = model.coef_[0]  # коэффициент наклона (β1)

# Выводим уравнение линейной регрессии
print(f"Уравнение линейной регрессии: Temperature = {intercept:.2f} + {slope:.2f} * CO2")

# Прогнозируем температуру на основе концентрации CO2
y_pred = model.predict(X)

# Визуализируем результаты
plt.scatter(df['CO2'], df['Temperature'], color='blue', label='Данные')
plt.plot(df['CO2'], y_pred, color='red', label='Линейная регрессия')
plt.title('Модель парникового эффекта')
plt.xlabel('Концентрация CO2 (ppm)')
plt.ylabel('Средняя температура на Земле (°C)')
plt.legend()
plt.grid(True)
plt.show()
