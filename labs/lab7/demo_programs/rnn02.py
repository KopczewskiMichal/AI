import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN


def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, 
                        activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

demo_model = create_RNN(2, 1, (3,1), activation=['linear', 'linear'])

# * Pobieranie wag z modelu
wx = demo_model.get_weights()[0]
wh = demo_model.get_weights()[1]
bh = demo_model.get_weights()[2]
wy = demo_model.get_weights()[3]
by = demo_model.get_weights()[4]

print('wx = ', wx, ' wh = ', wh, ' bh = ', bh, ' wy =', wy, 'by = ', by)

x = np.array([1, 2, 3]) # * Bardzo prosty wektor danych wejściowych
# Reshape the input to the required sample_size x time_steps x features 
x_input = np.reshape(x,(1, 3, 1))
y_pred_model = demo_model.predict(x_input)

# * Ręczne obliczanie wyniku
m = 2
h0 = np.zeros(m)
h1 = np.dot(x[0], wx) + h0 + bh
h2 = np.dot(x[1], wx) + np.dot(h1,wh) + bh
h3 = np.dot(x[2], wx) + np.dot(h2,wh) + bh
o3 = np.dot(h3, wy) + by

print('h1 = ', h1,'h2 = ', h2,'h3 = ', h3)

print("Prediction from network ", y_pred_model)
print("Prediction from our computation ", o3)


# wx =  [[-0.36795712 -0.5099192 ]]  wh =  [[ 0.11065364  0.99385905]
#  [ 0.99385905 -0.11065364]]  bh =  [0. 0.]  wy = [[-0.94251007]
#  [-1.2365237 ]] by =  [0.]
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 181ms/step
# h1 =  [[-0.36795712 -0.50991923]] h2 =  [[-1.28341786 -1.32911154]] h3 =  [[-2.56683574 -2.65822311]]
# Prediction from network  [[5.7062244]]
# Prediction from our computation  [[5.70622453]]