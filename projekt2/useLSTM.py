import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

LOOK_BACK = 3

model = load_model(f"LSTMmodel{LOOK_BACK}.keras")
TICKER = 'CSCO'

# Function to create dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

new_datafile = 'resources/stock_data.csv'
new_dataframe = pd.read_csv(new_datafile, engine='python')


new_dataframe = new_dataframe[["Date", TICKER]]
new_dataframe = new_dataframe.rename(columns={TICKER: "Close"})


print(new_dataframe.head())

new_close_values = new_dataframe['Close'].values

scaler = MinMaxScaler(feature_range=(0, 1))
new_close_values = new_close_values.reshape(-1, 1)
new_close_values_scaled = scaler.fit_transform(new_close_values)

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

newX, newY = create_dataset(new_close_values_scaled, LOOK_BACK)
newX = np.reshape(newX, (newX.shape[0], 1, newX.shape[1]))

newPredict = model.predict(newX)

newPredict = scaler.inverse_transform(newPredict)
newY = scaler.inverse_transform(newY.reshape(-1, 1))


plt.figure(figsize=(10, 6))
plt.plot(pd.to_datetime(new_dataframe['Date']),
         scaler.inverse_transform(new_close_values_scaled),
         label='Original Data')
plt.plot(pd.to_datetime(new_dataframe['Date'][LOOK_BACK:LOOK_BACK + len(newPredict)]),
         newPredict,
         label='Predicted Data')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()
plt.legend()
plt.savefig(f"docs/plots/{TICKER}_predicted_prices{LOOK_BACK}.png")
plt.show()
