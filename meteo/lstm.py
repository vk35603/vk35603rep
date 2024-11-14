import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv("D:\lstm_data.csv")  # Замените на путь к вашим данным
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Нормализация данных
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['Temperature']])  # Предполагается, что есть столбец "Temperature"

# Подготовка данных для LSTM
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(data_scaled, seq_length)

# Разделение данных на обучающие и тестовые наборы
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Изменение формы данных для LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Создание модели LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(X_train, y_train, batch_size=1, epochs=10)

# Прогнозирование на тестовом наборе
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Восстановление значений для тестовой выборки
y_test_rescaled = scaler.inverse_transform([y_test])

# Визуализация результатов
train = data[:train_size + seq_length]
valid = data[train_size + seq_length:]
valid['Predictions'] = predictions

plt.figure(figsize=(10, 10))
plt.title('Прогноз температуры воздуха с использованием LSTM')
plt.xlabel('Дата')
plt.ylabel('Температура (°C)')
plt.plot(train['Temperature'], label='Обучение')
plt.plot(valid[['Temperature']], label='Истинные значения')
plt.plot(valid[['Predictions']], label='Прогнозы')
plt.legend()
plt.show()
