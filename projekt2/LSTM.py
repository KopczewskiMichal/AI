import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ticker = "AAPL"

# Import danych treningowych
dataset_train = pd.read_csv('resources/new_data.csv')
dataset_train = dataset_train[dataset_train['ticker'] == ticker]

# Podział danych na treningowe i testowe
training_set = dataset_train.iloc[:-30, 1:2].values  # wszystkie miesiące z wyjątkiem ostatniego
test_set = dataset_train.iloc[-30:, 1:2].values  # ostatni miesiąc

# Manualne skalowanie danych treningowych
min_price_train = np.min(training_set)
max_price_train = np.max(training_set)
training_set_scaled = (training_set - min_price_train) / (max_price_train - min_price_train)

# Manualne skalowanie danych testowych
min_price_test = np.min(test_set)
max_price_test = np.max(test_set)
test_set_scaled = (test_set - min_price_test) / (max_price_test - min_price_test)

# Utwórz dane treningowe X_train, y_train
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)



# Przygotowanie danych testowych
X_test = []
for i in range(60, len(test_set_scaled)):
    X_test.append(test_set_scaled[i-60:i, 0])
X_test = np.array(X_test)

# Sprawdzenie kształtu danych testowych
if X_test.ndim == 2:
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Inicjalizacja modelu RNN
regressor = Sequential()

# Dodanie warstw LSTM i Dropout
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Dodanie kolejnych warstw LSTM i Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Dodanie warstwy wyjściowej
regressor.add(Dense(units=1))

# Kompilacja modelu RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Dopasowanie modelu do danych treningowych
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Przygotowanie danych testowych

# Przewidywanie cen akcji dla danych testowych
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = predicted_stock_price * (max_price_test - min_price_test) + min_price_test

# Wykres wyników
plt.plot(test_set, color='red', label=f'Real {ticker} Stock Price')
plt.plot(predicted_stock_price, color='blue', label=f'Predicted {ticker} Stock Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(f'{ticker} Stock Price')
plt.legend()

plt.savefig("docs/plot2.png")
