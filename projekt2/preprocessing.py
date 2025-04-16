import random

import numpy as np
import matplotlib.pyplot as plt
import pandas
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model



def predictLSTM(df: pandas.DataFrame) -> dict:
    print(df.head())
    LOOK_BACK = 3
    tickers = df.columns.drop('Date')


    model = load_model(f"LSTMmodel{LOOK_BACK}.keras")
    scaler = MinMaxScaler(feature_range=(0, 1))
    result = {}
    for act_ticker in tickers:
        print(len(result))
        close_values = df[act_ticker].values
        # Now you can work with the close_values for each ticker
        # print(f"Ticker: {act_ticker}, Values: {close_values}")


        close_values = close_values.reshape(-1, 1)  # Reshape to 2D
        close_values_scaled = scaler.fit_transform(close_values)
        # Predykcja zmiany w kolejnym dniu
        recent_close_values_scaled = close_values_scaled[-LOOK_BACK:]
        newX = np.reshape(recent_close_values_scaled, (1, 1, LOOK_BACK))
        newPredict = model.predict(newX)
        newPredict = scaler.inverse_transform(newPredict)
        tomorrow_prediction = newPredict[0][0]
        result[act_ticker] = tomorrow_prediction / close_values[-1]

    return result

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


if __name__ == '__main__':
    datafile = 'resources/stock_data.csv'
    df = read_csv(datafile, engine='python')
    predictions = predictLSTM(df)
    print(predictions)