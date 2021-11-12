import config

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import models, Input
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model, Model

from tensorflow.keras.datasets import mnist, fashion_mnist

# ---------------------------------------------------------------------------- #
# Sources: My code is built on the sources list below

# TF's tutorial on autoencoder
# https://www.tensorflow.org/tutorials/generative/
# Keras' tutorial on autoencoder
# https://blog.keras.io/building-autoencoders-in-keras.html

# These blog posts
# https://medium.com/analytics-vidhya/image-anomaly-detection-using-autoencoders-ae937c7fd2d1
# https://medium.com/analytics-vidhya/unsupervised-learning-and-convolutional-autoencoder-for-image-anomaly-detection-b783706eb59e
# https://towardsdatascience.com/anomaly-detection-using-autoencoders-5b032178a1ea
# https://towardsdatascience.com/a-keras-based-autoencoder-for-anomaly-detection-in-sequences-75337eaed0e5

# Chapter 8 of Chollet
# WEEK10 codes from Prof Hickman

# ---------------------------------------------------------------------------- #

n_bottleneck = 100
epochs = 100
batch_size = 1000
# Functional API representation of Prof Hickman's deep model in MNIST-DAE
# The code is borrowed from https://blog.keras.io/building-autoencoders-in-keras.html

# This is our input image
input_img = Input(shape=(28*28,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(n_bottleneck, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(28*28, activation='sigmoid')(decoded)

# This model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# This model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# ---------------------------------------------------------------------------- #
autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics = ['acc'])

# ---------------------------------------------------------------------------- #
# Validation is done on MNIST data too
(x_train, _), (x_val, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_val = x_val.reshape((len(x_val), np.prod(x_val.shape[1:])))

print("Normal MNIST train shape: ", x_train.shape)
print("Normal MNIST test shape: ", x_val.shape)


history = autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_val, x_val),
                verbose = 2)

# ---------------------------------------------------------------------------- #
def report(history,title='',I_PLOT=True):
    if(I_PLOT):
        #PLOT HISTORY
        epochs = range(1, len(history.history['loss']) + 1)
        plt.figure()
        plt.plot(epochs, history.history['loss'], 'r-', label='Training loss')
        plt.plot(epochs, history.history['val_loss'], 'b-', label='Validation loss')
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Binary Crossentropy")
        plt.legend()
        plt.savefig('./Plots/HW6.1_autoencoder_history_loss.png')
        plt.clf()
        
        plt.plot(epochs, history.history['acc'], 'ro', label='Training acc')
        plt.plot(epochs, history.history['val_acc'], 'b-', label='Validation acc')
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('./Plots/HW6.1_autoencoder_history_acc.png')
        plt.close()

# Save the autoencoder plots
report(history)
# Save the autoencoder
autoencoder.save('./Models/HW6.1_autoencoder.hdf5')

print("Finished training")

# ---------------------------------------------------------------------------- #
# Read in the MNIST FASION 
# This part is adapted from 
# http://www.renom.jp/notebooks/tutorial/clustering/anomaly-detection-using-autoencoder/notebook.html
print("Reading in Fashion MNIST")
(_, _), (x_test, _) = fashion_mnist.load_data()

# Prepare the data
x_test= x_test.astype('float32') / 255.
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print("Fashion MNIST shape: ", x_test.shape)

# ---------------------------------------------------------------------------- #
# Compare the autoencoder on normal held out MNIST and fashion MNIST

encoded_fashion = encoder.predict(x_test)
decoded_fashion = autoencoder.predict(x_test)

encoded_number = encoder.predict(x_val)
decoded_number = autoencoder.predict(x_val)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_val[i].reshape(28, 28))
  plt.title("original number")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_number[i].reshape(28, 28))
  plt.title("reconstructed number")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)      
  
plt.savefig('./Plots/HW6.1_reconstruct_number.png')
# plt.show()


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i].reshape(28, 28))
  plt.title("original fashion")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_fashion[i].reshape(28, 28))
  plt.title("reconstructed fashion")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.savefig('./Plots/HW6.1_reconstruct_fashion.png')
# plt.show()

# ---------------------------------------------------------------------------- #
# Define the anomaly threshold

x_train_pred = autoencoder.predict(x_train)
print("\nReconstruction loss is the binary crossentropy from reconstruction the training images")
reconstruction_loss = losses.binary_crossentropy(x_train, x_train_pred)

mean_loss = np.mean(reconstruction_loss)
print("Mean binary_crossentropy: ",mean_loss)
std_loss = np.std(reconstruction_loss)
print("Standard deviation: ", std_loss)
threshold = mean_loss+3*std_loss
print("Anomaly threshold: 3 std from the mean binary crossentropy: ", threshold)

# ---------------------------------------------------------------------------- #
# Find the anomaly rate for the normal MNIST validation
train_reconstruction_loss = np.asarray(reconstruction_loss)
train_anomalies = [i for i, loss in enumerate(train_reconstruction_loss) if loss>threshold]
train_anomalies_percent = len(train_anomalies)/len(train_reconstruction_loss)*100

print("\nTrain MNIST shape: ", x_train.shape)
print("Train MNIST anomaly detection rate: ", train_anomalies_percent, "%")

# ---------------------------------------------------------------------------- #
# Find the anomaly rate for the normal MNIST validation
x_val_pred = autoencoder.predict(x_val)
number_loss = losses.binary_crossentropy(x_val, x_val_pred)
number_loss = np.asarray(number_loss)
number_anomalies = [i for i, loss in enumerate(number_loss) if loss>threshold]
number_anomalies_percent = len(number_anomalies)/len(number_loss)*100

print("\nValidation MNIST shape: ", x_val.shape)
print("Validation MNIST anomaly detection rate: ", number_anomalies_percent, "%")


# ---------------------------------------------------------------------------- #
# Find the anomaly rate for the fashion MNIST set
x_test_pred = autoencoder.predict(x_test)
fashion_loss = losses.binary_crossentropy(x_test, x_test_pred)
fashion_loss = np.asarray(fashion_loss)
fashion_anomalies = [i for i, loss in enumerate(fashion_loss) if loss>threshold]
fashion_anomalies_percent = len(fashion_anomalies)/len(fashion_loss)*100

print("\nFashion MNIST test shape: ", x_test.shape)
print("Fashion MNIST anomaly detection rate: ", fashion_anomalies_percent, "%")

# ---------------------------------------------------------------------------- #
# Plot of the cross entropies and anomaly threshold

fig, ax = plt.subplots()
ax.axhline(y=threshold, color='k', linestyle=':')
ax.plot(range(len(fashion_loss)), fashion_loss, 'bo', alpha = 0.5, ms=1, label = "Fashion MNIST")
ax.plot(range(len(number_loss)), number_loss, 'ro', alpha = 0.5,  ms=1, label = "MNIST")
plt.title("Anomaly detection based on cross entropy")
plt.xlabel("Index")
plt.ylabel("Cross entropy between original and reconstructed image")
plt.legend()
plt.savefig("./Plots/HW6.1_cross_entropy_detection.png")
# plt.show()