import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

LOOK_BACK = 10
EPOCHS = 25

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
tf.random.set_seed(7)

# load the dataset
dataframe = read_csv('resources/SP500_history.csv', engine='python')
print(dataframe.columns)

# Konwersja kolumny 'Datetime' na typ daty
dataframe['Datetime'] = pd.to_datetime(dataframe['Date'])

# Wyodrębnienie dat i zamknięć
dates = dataframe['Datetime'].values
close_values = dataframe['Close'].values

# Skalowanie wartości zamknięcia
scaler = MinMaxScaler(feature_range=(0, 1))
close_values = close_values.reshape(-1, 1)  # Reshape do 2D
close_values_scaled = scaler.fit_transform(close_values)

# Konwersja dat do numerycznego formatu (np. liczba dni od pewnej daty)
dates_num = np.array([pd.Timestamp(d).timestamp() for d in dates]).reshape(-1, 1)
scaler_dates = MinMaxScaler(feature_range=(0, 1))
dates_scaled = scaler_dates.fit_transform(dates_num)

# Split into train and test sets
train_size = int(len(close_values_scaled) * 0.8)
test_size = len(close_values_scaled) - train_size
train_dates, test_dates = dates_scaled[0:train_size], dates_scaled[train_size:len(dates_scaled)]
train_close, test_close = close_values_scaled[0:train_size], close_values_scaled[train_size:len(close_values_scaled)]

# Reshape into X=t and Y=t+1
trainX, trainY = create_dataset(train_close, LOOK_BACK)
testX, testY = create_dataset(test_close, LOOK_BACK)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Early stopping callback
earlyStopCallback = EarlyStopping(monitor='loss', patience=3)

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(trainX.shape[1], LOOK_BACK)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=EPOCHS, batch_size=1, verbose=1, callbacks=[earlyStopCallback])
model.save(f"LSTMmodel{LOOK_BACK}.keras")

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY.reshape(-1, 1))

# Calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY, testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

# Shift train predictions for plotting
trainPredictPlot = np.empty_like(close_values_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[LOOK_BACK:len(trainPredict) + LOOK_BACK, :] = trainPredict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(close_values_scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (LOOK_BACK * 2) + 1:len(close_values_scaled) - 1, :] = testPredict

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(close_values_scaled), label='Original Data')
plt.plot(trainPredictPlot, label='Train Predict')
plt.plot(testPredictPlot, label='Test Predict')
plt.legend()
plt.savefig(f"docs/plots/lstm_plot{LOOK_BACK}.png")
