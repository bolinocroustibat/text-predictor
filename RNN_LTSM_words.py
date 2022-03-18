import json
import random
import string
from typing import Generator, Tuple

import gensim
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import typer
from gensim.models import Word2Vec
from keras.callbacks import LambdaCallback
from keras.layers import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.data_utils import get_file

from config import DATA_DIR
from helpers import CustomLogger

logger = CustomLogger()

# Load text data
with open(f"data/{DATA_DIR}/input.txt", "r") as f:
    lines = f.readlines()

sentences: list = [[word for word in line.lower().split()] for line in lines]

logger.info(f"Number of sentences: {len(sentences)}")


logger.info("Training word2vec...")
word_model = gensim.models.Word2Vec(
    sentences, vector_size=100, min_count=1, window=5, workers=100
)
logger.debug(word_model.wv)
pretrained_weights = word_model.wv.vectors
vocab_size, emdedding_size = pretrained_weights.shape
logger.info(f"Result embedding shape: {pretrained_weights.shape}")
logger.info("Checking similar words:")
for word in ["model", "network", "train", "learn"]:
    most_similar: list =[(similar, dist) for similar, dist in word_model.wv.most_similar(word)[:8]]
    logger.info(f"{word} -> {most_similar}")


def word2idx(word):
    return word_model.wv.vocab[word].index


def idx2word(idx):
    return word_model.wv.index2word[idx]


logger.info("Preparing the data for LSTM...")
train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
train_y = np.zeros([len(sentences)], dtype=np.int32)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence[:-1]):
        train_x[i, t] = word2idx(word)
    train_y[i] = word2idx(sentence[-1])
logger.info("train_x shape:", train_x.shape)
logger.info("train_y shape:", train_y.shape)

# logger.info("Training LSTM...")
# model = Sequential()
# model.add(
#     Embedding(
#         input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]
#     )
# )
# model.add(LSTM(units=emdedding_size))
# model.add(Dense(units=vocab_size))
# model.add(Activation("softmax"))
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")


# def sample(preds, temperature=1.0):
#     if temperature <= 0:
#         return np.argmax(preds)
#     preds = np.asarray(preds).astype("float64")
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)


# def generate_next(text, num_generated=10):
#     word_idxs = [word2idx(word) for word in text.lower().split()]
#     for i in range(num_generated):
#         prediction = model.predict(x=np.array(word_idxs))
#         idx = sample(prediction[-1], temperature=0.7)
#         word_idxs.append(idx)
#     return " ".join(idx2word(idx) for idx in word_idxs)


# def on_epoch_end(epoch, _):
#     logger.info("Generating text after epoch: %d" % epoch)
#     texts = [
#         "deep convolutional",
#         "simple and effective",
#         "a nonconvex",
#         "a",
#     ]
#     for text in texts:
#         sample = generate_next(text)
#         logger.info("%s... -> %s" % (text, sample))


# model.fit(
#     train_x,
#     train_y,
#     batch_size=128,
#     epochs=20,
#     callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)],
# )
