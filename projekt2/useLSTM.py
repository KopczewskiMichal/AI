import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

LOOK_BACK = 3

# Load the saved model
model = load_model(f"LSTMmodel{LOOK_BACK}.keras")
TICKER = 'AAPL'

# Function to create dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Load new data for prediction
new_datafile = 'resources/new_data.csv'  # Replace with your new CSV file path
new_dataframe = read_csv(new_datafile, engine='python')

# Filter for specific ticker
new_dataframe = new_dataframe.loc[new_dataframe['ticker'] == TICKER]

# Extract close values
new_close_values = new_dataframe['Close'].values

# Scale close values
scaler = MinMaxScaler(feature_range=(0, 1))
new_close_values = new_close_values.reshape(-1, 1)  # Reshape to 2D
new_close_values_scaled = scaler.fit_transform(new_close_values)

newX, newY = create_dataset(new_close_values_scaled, LOOK_BACK)

newX = np.reshape(newX, (newX.shape[0], 1, newX.shape[1]))

newPredict = model.predict(newX)

# Invert predictions
newPredict = scaler.inverse_transform(newPredict)
newY = scaler.inverse_transform(newY.reshape(-1, 1))

print(type(newY))
print(newY)

# Plot predictions
plt.plot(scaler.inverse_transform(new_close_values_scaled), label='Original Data')
plt.plot(range(LOOK_BACK, LOOK_BACK + len(newPredict)), newPredict, label='Predicted Data')
plt.legend()
plt.savefig(f"docs/plots/AAPL_predicted_prices{LOOK_BACK}.png")
plt.show()
