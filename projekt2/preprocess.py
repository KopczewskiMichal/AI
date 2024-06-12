import numpy as np
import matplotlib.pyplot as plt
import pandas
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model



def predictLSTM(df: pandas.DataFrame) -> dict:
    LOOK_BACK = 3
    tickers = df["ticker"].unique()

    model = load_model(f"LSTMmodel{LOOK_BACK}.keras")
    scaler = MinMaxScaler(feature_range=(0, 1))
    result = {}
    for act_ticker in tickers:
        new_dataframe = df.loc[df['ticker'] == act_ticker]
        close_values = new_dataframe['Close'].values
        close_values = close_values.reshape(-1, 1)  # Reshape to 2D
        close_values_scaled = scaler.fit_transform(close_values)
        # Ważnym aby dokonywać predykcji tylko na kolejny dzień.
        recent_close_values_scaled = close_values_scaled[-LOOK_BACK:]
        newX = np.reshape(recent_close_values_scaled, (1, 1, LOOK_BACK))
        newPredict = model.predict(newX)
        # Invert prediction
        newPredict = scaler.inverse_transform(newPredict)
        tomorrow_prediction = newPredict[0][0]
        result[act_ticker] = tomorrow_prediction / close_values[-1]
        # Print the prediction for the next day
        # print(f"Predicted {act_ticker} close price for the next day:", tomorrow_prediction)

    return result


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


if __name__ == '__main__':
    datafile = 'resources/new_data.csv'  # Replace with your new CSV file path
    df = read_csv(datafile, engine='python')
    predictions = predictLSTM(df)
    print(predictions)