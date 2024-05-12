# Load LSTM network and generate text
import sys
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
tokenized_text = wordpunct_tokenize(raw_text)
tokens = sorted(list(dict.fromkeys(tokenized_text)))

#print("Tokens: ")
#print(tokens)
tok_to_int = dict((c, i) for i, c in enumerate(tokens))
int_to_tok = dict((i, c) for i, c in enumerate(tokens))
#print("TokensToNumbers: ")
#print(tok_to_int)

# summarize the loaded data
n_tokens = len(tokenized_text)
n_token_vocab = len(tokens)
print("Total Tokens: ", n_tokens)
print("Unique Tokens (Token Vocab): ", n_token_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_tokens - seq_length, 1):
	seq_in = tokenized_text[i:i + seq_length]
	seq_out = tokenized_text[i + seq_length]
	dataX.append([tok_to_int[tok] for tok in seq_in])
	dataY.append(tok_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_token_vocab)
# one hot encode the output variable
y = to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
filename = "big-token-model.keras"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ' '.join([int_to_tok[value] for value in pattern]), "\"")
# generate tokens
print("Generated text:")
for i in range(100):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_token_vocab)
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = int_to_tok[index]
	seq_in = [int_to_tok[value] for value in pattern]
	sys.stdout.write(result+" ")
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")


#* wynik po 5 epokach treningu
# Total Tokens:  38030
# Unique Tokens (Token Vocab):  3236
# Total Patterns:  37930
# /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/keras/src/layers/rnn/rnn.py:204: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
#   super().__init__(**kwargs)
# Seed:
# " fact , we went to school every day —” “ _i ’ ve_ been to a day - school , too ,” said alice ; “ you needn ’ t be so proud as all that .” “ with extras ?” asked the mock turtle a little anxiously . “ yes ,” said alice , “ we learned french and music .” “ and washing ?” said the mock turtle . “ certainly not !” said alice indignantly . “ ah ! then yours wasn ’ t a really good school ,” said the mock turtle in a tone of "
# Generated text:
# , “ the , “ , “ , “ , “ , “ , “ , “ , “ , “ , , “ the , “ , , “ the , “ , , “ the , “ , , “ the , “ , , “ the , “ , , “ the , “ , , “ the , “ , , “ the , “ , , “ the , “ , , “ the , “ , , “ the , “ , , “ the , “ , , “ the , “ ,
# Done.

#* po bardzo małej ilości epok i krótkim treningu rezultat nie przypomina niczego sensownego ale zwraca istniejące słowo.