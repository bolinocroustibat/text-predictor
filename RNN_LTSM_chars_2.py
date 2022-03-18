from datetime import datetime
import sys

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from config import DATA_DIR
from helpers import CustomLogger


logger = CustomLogger()


# Load text data
with open(f"data/{DATA_DIR}/input.txt", "r") as f:
    text: str = f.read()
text = text.lower()

# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(text)
n_vocab = len(chars)
logger.info(f"Total characters: {n_chars}")
logger.info(f"Total unique chars (vocab): {n_vocab}")

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = text[i:i + seq_length]
	seq_out = text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
logger.info(f"Total patterns: {n_patterns}")

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
# model = Sequential()
# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
# model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the LSTM model (better)
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# display the shape of the model
model.summary()

# # save the model with its weights
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)

# # load the network weights
# filename = "weights-improvement-19-1.9435.hdf5"
# model.load_weights(filename)

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
logger.info("Seed:")
logger.info([int_to_char[value] for value in pattern])

# generate characters
sequence: str = ""
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	logger.success(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
	sequence.add(result)

out_filename: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
with open(f"data/{DATA_DIR}/{out_filename}", "w") as outfile:
	outfile.write(sentence)

logger.success("Done.")
