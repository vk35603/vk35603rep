import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Загрузка данных
# Убедитесь, что путь к файлу правильный
data = pd.read_csv(r'D:\arima_data.csv')  # используем raw строку для пути
data['date'] = pd.to_datetime(data['date'])  # Преобразование столбца 'date' в datetime
data.set_index('date', inplace=True)

# Проверка на пропущенные значения
data = data.dropna()  # Убираем строки с пропущенными значениями

# Разделение данных на обучающую и тестовую выборки
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Настройка параметров ARIMA модели (p, d, q)
p = 1  # порядок авторегрессии
d = 1  # порядок разности
q = 1  # порядок скользящего среднего

# Обучение модели ARIMA
model = ARIMA(train['value'], order=(p, d, q))
fitted_model = model.fit()

# Прогнозирование
forecast_steps = len(test)  # Прогнозируем на количество шагов в тестовой выборке
forecast = fitted_model.forecast(steps=forecast_steps)
test['forecast'] = forecast

# Оценка точности
mse = mean_squared_error(test['value'], test['forecast'])
print(f'Mean Squared Error: {mse}')

# Визуализация результатов
plt.figure(figsize=(10, 10))
plt.plot(train['value'], label='Обучающая выборка')
plt.plot(test['value'], label='Прогноз')
plt.plot(test['forecast'], label='Прогноз ARIMA', linestyle='--')

plt.xlabel("Годы от начальной даты (1821 г.)")
plt.ylabel("Температура (°C)")
plt.title("Среднегодовая температура в июне (Москва)")

plt.legend()
plt.show()
