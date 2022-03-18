import json
import random
from typing import Generator, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import typer
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

from config import DATA_DIR
from helpers import CustomLogger

logger = CustomLogger()

# Load text data
with open(f"data/{DATA_DIR}/input.txt", "r") as f:
    text: str = f.read()
text = text.lower()
# logger.debug(text[:500])

# Map each char to int
vocab = set(text)
vocab_size: int = len(vocab)
vocab_to_int: dict = {l: i for i, l in enumerate(vocab)}
int_to_vocab: dict = {i: l for i, l in enumerate(vocab)}

# Text vectorization
encoded: list = [vocab_to_int[l] for l in text]
inputs, targets = encoded, encoded[1:]


def generate_batches(
    inputs: list, targets: list, seq_len: int, batch_size: int, noise=0
) -> Tuple[Generator, Generator]:
    # Size of each chunk
    # Si le texte d'entraînement fait 20000 caractères, et qu'on entraine à chaque epoch avec 64 batches de data, chaque batch fera 200000//64=310 caractères.
    chunk_size: int = (len(inputs) - 1) // batch_size
    # Number of sequences per chunk
    sequences_per_chunk: int = chunk_size // seq_len

    for s in range(0, sequences_per_chunk):
        batch_inputs = np.zeros((batch_size, seq_len))
        batch_targets = np.zeros((batch_size, seq_len))
        for b in range(0, batch_size):
            fr = (b * chunk_size) + (s * seq_len)
            to = fr + seq_len
            batch_inputs[b] = inputs[fr:to]
            batch_targets[b] = inputs[fr + 1 : to + 1]

            if noise > 0:
                noise_indices = np.random.choice(seq_len, size=noise)
                batch_inputs[b][noise_indices] = np.random.randint(0, vocab_size)

        yield batch_inputs, batch_targets


# for batch_inputs, batch_targets in generate_batches(inputs, targets, 5, 32, noise=0):
#     logger.info(batch_inputs[0], batch_targets[0])
#     break

# for batch_inputs, batch_targets in generate_batches(inputs, targets, 5, 32, noise=3):
#     logger.info(batch_inputs[0], batch_targets[0])
#     break


class OneHot(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self.depth = depth

    def call(self, x, mask=None):
        return tf.one_hot(tf.cast(x, tf.int32), self.depth)


class RnnModel(tf.keras.Model):
    def __init__(self, vocab_size):
        super(RnnModel, self).__init__()
        # Convolutions
        self.one_hot = OneHot(vocab_size)

    def call(self, inputs):
        output = self.one_hot(inputs)
        return output


batch_inputs, batch_targets = next(
    generate_batches(inputs=inputs, targets=targets, seq_len=50, batch_size=32)
)

model = RnnModel(vocab_size)
output = model.predict(batch_inputs)

logger.debug(f"Shape of the output of the model: {output.shape}")

# logger.debug(output)

logger.debug(
    "Input letter is: {} ({})".format(
        batch_inputs[0][0], int_to_vocab[batch_inputs[0][0]]
    )
)
logger.debug("One hot representation of the letter: {}".format(output[0][0]))

# assert(output[int(batch_inputs[0][0])]==1)


### Set up the models, create the layers

# On envoie en input 64 batches de phrases de 50 caractères chacune

# Set the input of the model
tf_inputs = tf.keras.Input(shape=(None,), batch_size=64)
# Convert each value of the  input into a one encoding vector
one_hot = OneHot(vocab_size)(tf_inputs)
# Stack LSTM cells
rnn_layer1 = LSTM(128, return_sequences=True, stateful=True)(one_hot)
rnn_layer2 = LSTM(128, return_sequences=True, stateful=True)(rnn_layer1)
# Create the outputs of the model
hidden_layer = Dense(128, activation="relu")(rnn_layer2)
outputs = Dense(vocab_size, activation="softmax")(hidden_layer)

### Setup the model
model = tf.keras.Model(inputs=tf_inputs, outputs=outputs)

# Test the shape of the model
model.summary()

# model2 = Sequential()
# # Set the input of the model
# model2.add(tf.keras.Input(shape=(None,), batch_size=64))
# # Convert each value of the  input into a one encoding vector
# model2.add(OneHot(vocab_size))
# # Stack LSTM cells
# model2.add(LSTM(128, return_sequences=True, stateful=True))
# model2.add(LSTM(128, return_sequences=True, stateful=True))
# # Outputs of the model
# model2.add(Dense(128, activation="relu"))
# model2.add(Dense(vocab_size, activation="softmax"))
# model2.summary()

# Start by resetting the cells of the RNN
model.reset_states()

# Get one batch
batch_inputs, batch_targets = next(
    generate_batches(inputs=inputs, targets=targets, seq_len=50, batch_size=64)
)
logger.debug(f"Shape of the inputs: {batch_inputs.shape}")

# Make a first prediction
outputs = model.predict(batch_inputs)
first_prediction = outputs[0][0]

# Reset the states of the RNN states
model.reset_states()

# Make another prediction to check the difference
outputs = model.predict(batch_inputs)
second_prediction = outputs[0][0]

# Check if both prediction are equal
assert set(first_prediction) == set(second_prediction)


# Set the loss and objective
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Set some metrics to track the progress of the training
## Loss
train_loss = tf.keras.metrics.Mean(name="train_loss")
## Accuracy
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

# Set the train method and the predict method in graph mode
@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        # Make a prediction on all the batches
        predictions = model(inputs)
        # Get the error/loss on these predictions
        loss = loss_object(targets, predictions)
    # Compute the gradient which respect to the loss
    gradients = tape.gradient(loss, model.trainable_variables)
    # Change the weights of the model
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # The metrics are accumulate over time. You don't need to average it yourself.
    train_loss(loss)
    train_accuracy(targets, predictions)


@tf.function
def predict(inputs):
    # Make a prediction on all the batch
    predictions = model(inputs)
    return predictions


# Train the model

model.reset_states()

for epoch in range(4000):
    for batch_inputs, batch_targets in generate_batches(
        inputs=inputs, targets=targets, seq_len=50, batch_size=64, noise=13
    ):
        train_step(batch_inputs, batch_targets)
    template = "\r Epoch {}, Train Loss: {}, Train Accuracy: {}"
    print(
        template.format(epoch, train_loss.result(), train_accuracy.result() * 100),
        end="",
    )
    model.reset_states()


# Save the model

model.save("model_rnn.h5")

with open(f"data/{DATA_DIR}/model_rnn_vocab_to_int", "w") as f:
    f.write(json.dumps(vocab_to_int))
with open(f"data/{DATA_DIR}/model_rnn_int_to_vocab", "w") as f:
    f.write(json.dumps(int_to_vocab))


# Generate some text
model.reset_states()

size_poetries = 300

poetries = np.zeros((64, size_poetries, 1))
sequences = np.zeros((64, 100))
for b in range(64):
    rd = np.random.randint(0, len(inputs) - 100)
    sequences[b] = inputs[rd : rd + 100]

for i in range(size_poetries + 1):
    if i > 0:
        poetries[:, i - 1, :] = sequences
    softmax = predict(sequences)
    # Set the next sequences
    sequences = np.zeros((64, 1))
    for b in range(64):
        argsort = np.argsort(softmax[b][0])
        argsort = argsort[::-1]
        # Select one of the strongest 4 proposals
        sequences[b] = argsort[0]

for b in range(64):
    sentence: str = "".join([int_to_vocab[i[0]] for i in poetries[b]])
    logger.success(sentence)
    with open(f"data/{dataset_name}/{out_filename}.txt", "w") as outfile:
        outfile.write(sentence)
