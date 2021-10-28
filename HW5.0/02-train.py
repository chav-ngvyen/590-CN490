import config # config file

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_LOG_LEVEL

# print(os.environ['TF_CPP_MIN_LOG_LEVEL'])
# exit()

import numpy as np
np.random.seed(42)

import regex as re
from nltk import tokenize 


import tensorflow as tf 
import os.path

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Embedding
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

# ---------------------------------------------------------------------------- #
# import logging
# logging.basicConfig(filename='./Logs/log.txt',
#             filemode='w',
#             level=logging.INFO)
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
# Load data cleaned from 01-clean.py


print("\nData:")
x_train = np.load(config.processed_path+'x_train.npy'); print("X train shape: ", x_train.shape)
y_train = np.load(config.processed_path+'y_train.npy'); print("Y train shape: ", y_train.shape)
x_val = np.load(config.processed_path+'x_val.npy'); print("X val shape: ", x_val.shape)
y_val = np.load(config.processed_path+'y_val.npy'); print("Y val shape: ", y_val.shape)

#x_test = np.load("./Data/x_test.npy"); print("X test shape: ", x_test.shape)
#y_test = np.load("./Data/y_test.npy"); print("Y test shape: ", y_test.shape)

# ---------------------------------------------------------------------------- #
# Define models

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

# ---------------------------------------------------------------------------- #
# Compile the model
model.compile(optimizer=RMSprop(lr=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

# ---------------------------------------------------------------------------- #
# Create callbacks

checkpoint = ModelCheckpoint(filepath=config.best_model_path, 
                             monitor=config.SCORE_TO_MONITOR,
                             verbose=1, 
                             save_best_only=True,
                             mode=config.MODE)
CALLBACKS = [checkpoint]
# ---------------------------------------------------------------------------- #
# Fit model
history = model.fit(x_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose = 1,
                    validation_data = (x_val, y_val),
                    callbacks = CALLBACKS)
# ---------------------------------------------------------------------------- #
# Save history

print("\n Saving training scores as: ",config.best_model_training_scores_path)
np.save(config.best_model_training_scores_path,history.history)



exit()
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# print("\n Training Loss: ",history.history['loss'], "; Training Accuracy: ", history.history['acc'])
# print("\n Validation Loss: ",history.history['val_loss'], "; Validation Accuravy: ", history.history['val_acc'])

# # Save scores
# print("\n Saving scores as: ",config.model_save_path+'_training_scores.npy')
# np.save(config.model_save_path+'_training_scores.npy',history.history)


exit()
 
# mod_history=np.load('history.npy',allow_pickle='TRUE').item()
# print(mod_history)
# #model.save(config.model_save_path+'.h5')
# print("Saved model, exiting")
# exit()
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

