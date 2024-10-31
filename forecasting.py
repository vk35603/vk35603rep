import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Создаем небольшой временной ряд температуры
np.random.seed(0)
dates = pd.date_range(start='2023-01-01', end='2023-01-07', freq='h')
temperature = np.random.normal(loc=25, scale=5, size=len(dates))
df_temperature = pd.DataFrame({'Дата': dates, 'Температура': temperature})

# Устанавливаем 'Дата' в качестве индекса
df_temperature.set_index('Дата', inplace=True)

# Построим график временного ряда
plt.figure(figsize=(12, 4))
plt.plot(df_temperature.index, df_temperature['Температура'])
plt.title('Временной ряд температуры')
plt.xlabel('Дата и время')
plt.ylabel('Температура (°C)')
plt.grid(True)
plt.show()

# Рассчитываем автокорреляцию и частичную автокорреляцию
plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_acf(df_temperature['Температура'], lags=50, ax=plt.gca())
plt.title('Автокорреляция')

plt.subplot(212)
plot_pacf(df_temperature['Температура'], lags=50, ax=plt.gca())
plt.title('Частичная автокорреляция')

plt.tight_layout()
plt.show()