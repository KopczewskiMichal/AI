import numpy as np
from keras.layers import SimpleRNN

inputs = np.random.random([32, 10, 8]).astype(np.float32)
print("Inputs: ")
print(inputs)

simple_rnn = SimpleRNN(4)

output = simple_rnn(inputs)  # The output has shape `[32, 4]`.
print("Output: ")
print(output)

simple_rnn = SimpleRNN(
    4, return_sequences=True, return_state=True)

# whole_sequence_output has shape `[32, 10, 4]`.
# final_state has shape `[32, 4]`.
whole_sequence_output, final_state = simple_rnn(inputs)


# 2024-05-12 22:44:43.546116: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Inputs:
# [[[0.6133957  0.6036205  0.600268   ... 0.7134928  0.71699625 0.34252712]
#   [0.72535956 0.93291134 0.09493816 ... 0.94094265 0.8818395  0.9983714 ]
#   [0.53982973 0.8134416  0.21623242 ... 0.23708259 0.45150945 0.05229082]
#   ...
#   [0.669157   0.49719912 0.19289634 ... 0.16092977 0.97953385 0.38966617]
#   [0.10543093 0.9834885  0.8155254  ... 0.963143   0.465905   0.04953254]
#   [0.06197885 0.33467233 0.18199362 ... 0.5663574  0.44580665 0.3048611 ]]

#  [[0.90852106 0.5335378  0.30474105 ... 0.8472297  0.21837655 0.31371656]
#   [0.00758201 0.60898554 0.6614765  ... 0.53175306 0.15831243 0.96090686]
#   [0.7471171  0.496808   0.59963685 ... 0.9609048  0.33636224 0.18929133]
#   ...
#   [0.73395854 0.7782181  0.22896113 ... 0.24637334 0.07673361 0.35631078]
#   [0.03333575 0.33001056 0.5823729  ... 0.6685463  0.22310394 0.6058562 ]
#   [0.9263984  0.29243305 0.06776261 ... 0.15047292 0.49257404 0.76386595]]

#  [[0.25009736 0.3011825  0.68047875 ... 0.33665392 0.9454105  0.912507  ]
#   [0.30034497 0.94199306 0.15951228 ... 0.8867843  0.65439457 0.5942861 ]
#   [0.09665319 0.4652996  0.5768039  ... 0.63964254 0.44380412 0.07890639]
#   ...
#   [0.6522422  0.9400963  0.3960306  ... 0.92648536 0.5208738  0.49158534]
#   [0.5486321  0.38619572 0.15329367 ... 0.53248113 0.39672363 0.40950048]
#   [0.6408589  0.31570154 0.9187127  ... 0.92102623 0.08297544 0.41716084]]

#  ...

#  [[0.2014742  0.87840587 0.6887069  ... 0.94450486 0.1727269  0.6054538 ]
#   [0.562285   0.71196085 0.19024621 ... 0.02201548 0.8104511  0.8802205 ]
#   [0.83088326 0.5607456  0.96004015 ... 0.9485136  0.45266652 0.6375052 ]
#   ...
#   [0.11059068 0.92439115 0.6835951  ... 0.6955775  0.97770286 0.8985632 ]
#   [0.05918076 0.6886393  0.17849876 ... 0.77635974 0.29332477 0.1693684 ]
#   [0.97944456 0.43541104 0.8081337  ... 0.34141943 0.06828712 0.92190427]]

#  [[0.6588582  0.7585072  0.6132937  ... 0.07119424 0.9786859  0.22017755]
#   [0.1915317  0.7294521  0.44270414 ... 0.00658268 0.1747828  0.657666  ]
#   [0.6139039  0.6792326  0.6624624  ... 0.5593505  0.56669044 0.63780355]
#   ...
#   [0.08772446 0.22045389 0.1627578  ... 0.7842412  0.512374   0.92368907]
#   [0.06258297 0.3235114  0.22616613 ... 0.18366899 0.21503022 0.86280864]
#   [0.5365955  0.8004361  0.8961998  ... 0.19597378 0.2613641  0.45869622]]

#  [[0.8097364  0.03311673 0.6474315  ... 0.77999014 0.43863815 0.21817903]
#   [0.6955099  0.89531827 0.08043681 ... 0.06685235 0.7414596  0.18688631]
#   [0.37819877 0.2877197  0.08541992 ... 0.6331703  0.31756994 0.4897454 ]
#   ...
#   [0.5521306  0.2438023  0.19374394 ... 0.03836357 0.9542841  0.43441856]
#   [0.5366668  0.39358178 0.6761177  ... 0.2546202  0.5963845  0.92032176]
#   [0.33437392 0.4373795  0.03091921 ... 0.3092586  0.54938215 0.650283  ]]]
# Output:
# tf.Tensor(
# [[-7.8797919e-01  2.0614095e-01 -1.2148884e-02  4.8627573e-01]
#  [-1.5390454e-01  3.0917877e-01  9.7107053e-02  3.7130970e-01]
#  [ 9.4224143e-01 -3.7759755e-02 -5.2291404e-02  3.0230516e-01]
#  [ 4.3325153e-01 -1.7484950e-01 -7.0414335e-01  4.0923750e-01]
#  [ 7.3420262e-01  1.9125511e-01 -1.7917855e-01  3.1767285e-01]
#  [ 1.0299469e-01 -4.3491859e-02 -5.0019121e-01  3.2341111e-01]
#  [ 1.7915717e-01 -1.0467205e-02 -1.3639900e-02  7.1953690e-01]
#  [ 8.0564046e-01  4.4797501e-01  1.4940195e-01  6.0956305e-01]
#  [ 1.6005932e-01 -2.5128978e-01 -4.9756518e-01  9.1630363e-01]
#  [-1.2161138e-01 -7.3374110e-01 -1.9641696e-01  7.4796313e-01]
#  [-5.0091743e-02 -4.7329125e-01 -8.7237984e-01  6.2440914e-01]
#  [ 9.3537534e-04 -7.4701715e-01 -9.2243719e-01  4.7887349e-01]
#  [-3.9407748e-01 -2.2430146e-01 -2.7862203e-01  8.4586847e-01]
#  [-4.6674567e-01  2.2338060e-01  1.9433816e-01  7.3191959e-01]
#  [ 4.3981108e-01 -4.1488943e-01 -7.4342120e-01  7.0754838e-01]
#  [ 2.1073537e-01 -4.4125959e-01 -1.8389121e-01  7.7030373e-01]
#  [ 3.0080423e-01 -2.3105630e-01 -7.3919773e-01  8.9478177e-01]
#  [ 9.1225185e-02 -1.3784187e-01 -2.7300587e-01  8.3138216e-01]
#  [-5.0638431e-01 -8.0500054e-01  2.7181855e-01  5.1115078e-01]
#  [ 7.9191083e-01 -3.9676271e-02 -7.1943253e-01  4.8204586e-01]
#  [ 3.8173795e-04  5.1526171e-01  4.3289840e-01  8.4730744e-01]
#  [ 5.6264967e-01  1.7149851e-01 -7.7296561e-01  1.3755685e-01]
#  [ 1.3047920e-01 -1.5854748e-01  5.2407789e-01  7.2268152e-01]
#  [ 4.9173504e-01 -6.0196239e-01 -8.9052761e-01  5.0322372e-01]
#  [ 5.0837576e-02 -6.7191726e-01 -6.3193756e-01  3.7554559e-01]
#  [ 4.2823336e-01  2.3846029e-01  3.4443402e-01  3.0855915e-01]
#  [ 5.9568751e-01 -3.7990922e-01 -1.9960137e-01  4.4958889e-01]
#  [ 2.0983341e-01  3.9143220e-02 -4.8723075e-01  4.2710912e-01]
#  [ 3.4258664e-01 -2.0359921e-01 -1.3851720e-01  2.7688128e-01]
#  [ 7.6681250e-01  1.5681800e-01 -9.6030869e-02  7.2647882e-01]
#  [ 2.7786314e-01 -7.6008958e-01 -7.3849100e-01  6.1266929e-01]
#  [-5.4925275e-01  3.7359229e-01 -6.3409545e-03  2.8572181e-01]], shape=(32, 4), dtype=float32)