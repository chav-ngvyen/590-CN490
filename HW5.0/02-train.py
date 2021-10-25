import tensorflow as tf 
import numpy as np
np.random.seed(42)

import regex as re
from nltk import tokenize 
import os
import os.path

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Embedding
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

import config # config file

# ---------------------------------------------------------------------------- #
import logging
logging.basicConfig(filename='./Logs/log.txt',
            filemode='w',
            level=logging.INFO)
# ---------------------------------------------------------------------------- #
# Print config details
print("\nConfig:")
print("\nTest: ", config.TEST)
maxlen = config.maxlen; print("Max length: ", maxlen)
chunk_size = config.chunk_size; print("Sentences per chunk: ", chunk_size)
max_words = config.max_words; print("Max words: ", max_words)
training_split = config.training_split; print("Training split: ", training_split)

print("\nHyper-parameters")
EPOCHS=config.EPOCHS; print("Epochs: ", EPOCHS)
BATCH_SIZE=config.BATCH_SIZE; print("Batch size: ", BATCH_SIZE) 
MODEL = config.MODEL; print("Model: ", MODEL)
# ---------------------------------------------------------------------------- #
# Load data cleaned by 01-clean.py
print("\nData:")
x_train = np.load("./Data/x_train.npy"); print("X train shape: ", x_train.shape)
y_train = np.load("./Data/y_train.npy"); print("Y train shape: ", y_train.shape)
x_val = np.load("./Data/x_val.npy"); print("X val shape: ", x_val.shape)
y_val = np.load("./Data/y_val.npy"); print("Y val shape: ", y_val.shape)
x_test = np.load("./Data/x_test.npy"); print("X test shape: ", x_test.shape)
y_test = np.load("./Data/y_test.npy"); print("Y test shape: ", y_test.shape)

# ---------------------------------------------------------------------------- #
# Fit model

# This part should get re-factored using last week's submission
if (MODEL == "CONV1D" ):
    model = Sequential()
    model.add(layers.Embedding(max_words, BATCH_SIZE, input_length=maxlen))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(3, activation = 'softmax'))
    model.summary()

if (MODEL == "SRNN"):
    model = Sequential()
    model.add(layers.Embedding(max_words, BATCH_SIZE))
    model.add(layers.SimpleRNN(BATCH_SIZE))
    model.add(layers.Dense(3, activation='softmax'))
    model.summary()

if (MODEL == "LSTM"):
    model = Sequential()
    model.add(layers.Embedding(max_words, BATCH_SIZE))
    model.add(layers.LSTM(BATCH_SIZE))
    model.add(layers.Dense(3, activation = 'softmax'))
    model.summary()

model.compile(optimizer=RMSprop(lr=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                   epochs=EPOCHS,
                    batch_size=128,
                    verbose = 1,
                    validation_data = (x_val, y_val))
print(history)

model.save(config.model_save_path+'.h5')

# # # ---------------------------------------------------------------------------- #
# # # Plot it

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.savefig(config.plot_save_path+'_acc.png', dpi=300)
# plt.clf()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.savefig(config.plot_save_path+'_loss.png', dpi=300)

