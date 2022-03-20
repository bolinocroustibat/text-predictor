import sys
from datetime import datetime

import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils

from config import DATA_DIR, EPOCHS
from helpers import CustomLogger

logger = CustomLogger()


# Load text data
with open(f"data/{DATA_DIR}/input.txt", "r") as f:
    text: str = f.read()
text: str = text.lower()

# create mapping of unique chars to integers, and a reverse mapping
chars: list = sorted(list(set(text)))
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
    seq_in = text[i : i + seq_length]
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
model.add(Dense(y.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")

# display the shape of the model
model.summary()

# save the model with its weights
filepath = "weights_{epoch:02d}_{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath=f"models/{DATA_DIR}/"+filepath, monitor="loss", verbose=1, save_best_only=True, mode="min"
)
callbacks_list = [checkpoint]

# fit the model
model.fit(X, y, epochs=EPOCHS, batch_size=64, callbacks=callbacks_list)

# pick a random seed
start = numpy.random.randint(0, len(dataX) - 1)
pattern = dataX[start]
logger.info("Seed:")
logger.info([int_to_char[value] for value in pattern])

# generate characters
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    pattern.append(index)
    pattern = pattern[1 : len(pattern)]

    out_filename: str = f"data/{DATA_DIR}/{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
    with open(out_filename, "a") as outfile:
        outfile.write(result)

logger.success(f"Output file {out_filename} written.")
