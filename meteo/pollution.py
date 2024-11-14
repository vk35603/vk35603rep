import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Загрузка данных из CSV файла
data = pd.read_csv('D:\pollution_data.csv')

# Предположим, что в данных есть колонки: 'date', 'pollution_level', 'temperature', 'humidity'
# Преобразование даты в формат datetime
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Визуализация исходных данных
plt.figure(figsize=(10, 10))
plt.plot(data.index, data['pollution_level'], label='Уровень загрязнения')
plt.xlabel('Дата')
plt.ylabel('Уровень загрязнения, мкг/м³')
plt.title('Колебания уровня загрязнения воздуха')
plt.legend()
plt.show()

# Подготовка данных для модели
X = data[['temperature', 'humidity']]
y = data['pollution_level']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Визуализация предсказаний
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel('Фактический уровень загрязнения')
plt.ylabel('Предсказанный уровень загрязнения')
plt.title('Сравнение фактических и предсказанных уровней загрязнения')
plt.show()
