import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import models, Input
from tensorflow.keras import layers, losses
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model, Model

from tensorflow.keras.datasets import mnist, fashion_mnist

from sklearn.metrics import accuracy_score
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

epochs = 100
batch_size = 1000
# Functional API representation of Prof Hickman's deep model in MNIST-DAE
# The code is borrowed from https://blog.keras.io/building-autoencoders-in-keras.html

input_img = Input(shape=(28, 28, 1))
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)

decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
# x = layers.Conv2D(14, (3, 3), activation='relu', padding='same')(input_img)
# # x = layers.MaxPooling2D((2, 2), padding='same')(x)
# # x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# #x = layers.MaxPooling2D((2, 2), padding='same')(x)
# encoded = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# # encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# # "decoded" is the loss reconstruction of the input
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
# #x = layers.UpSampling2D((2, 2))(x)
# # x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# #x = layers.UpSampling2D((2, 2))(x)
# x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# #x = layers.UpSampling2D((2, 2))(x)
# decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)



autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

autoencoder.summary()
# ---------------------------------------------------------------------------- #
# Validation is done on MNIST data too
(x_train, _), (x_val, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_val = np.reshape(x_val, (len(x_val), 28, 28, 1))

print("Normal MNIST train shape: ", x_train.shape)
print("Normal MNIST test shape: ", x_val.shape)


history = autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_val, x_val),
                verbose = 1)

x_train_pred= autoencoder.predict(x_train)


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
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('./Plots/HW6.2_autoencoder_history_loss.png')
        plt.clf()
        
        plt.plot(epochs, history.history['acc'], 'ro', label='Training acc')
        plt.plot(epochs, history.history['val_acc'], 'b-', label='Validation acc')
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('./Plots/HW6.2_autoencoder_history_acc.png')
        plt.close()

# Save the autoencoder plots
report(history)
# Save the autoencoder
autoencoder.save('./Models/HW6.2_autoencoder.hdf5')

print("Finished training")
# exit()
# ---------------------------------------------------------------------------- #
# Read in the MNIST FASION 
# This part is adapted from 
# http://www.renom.jp/notebooks/tutorial/clustering/anomaly-detection-using-autoencoder/notebook.html
print("Reading in Fashion MNIST")
(_, _), (x_test, _) = fashion_mnist.load_data()

# Prepare the data
x_test= x_test.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

print("Fashion MNIST shape: ", x_test.shape)


# ---------------------------------------------------------------------------- #
# Compare the autoencoder on normal held out MNIST and fashion MNIST

decoded_fashion = autoencoder.predict(x_test)
decoded_number = autoencoder.predict(x_val)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_val[i].reshape(28, 28, 1))
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_number[i].reshape(28, 28,1))
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)      
  
plt.savefig('./Plots/HW6.2_reconstruct_MNIST.png')
# plt.show()


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i].reshape(28, 28))
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_fashion[i].reshape(28, 28))
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.savefig('./Plots/HW6.2_reconstruct_fashion_MNIST.png')
# plt.show()


# ---------------------------------------------------------------------------- #
# Define the anomaly threshold

x_train_pred = autoencoder.predict(x_train)


mse = losses.MeanSquaredError()
train_losses = []
for i in range(len(x_train)):
  train_loss = mse(x_train[i], x_train_pred[i]).numpy()
  train_losses.append(train_loss)

mean_loss = np.mean(train_losses)
print("Mean of MSE: ",mean_loss)
std_loss = np.std(train_losses)
print("Standard deviation: ", std_loss)
threshold = mean_loss+3*std_loss
print("Anomaly threshold: 3 std from the mean MSE: ", threshold)
max_loss = np.max(train_loss)
print("\nMax loss: ", max_loss)

train_anomalies = [i for i, loss in enumerate(train_losses) if loss>threshold]
train_anomalies_percent = len(train_anomalies)/len(train_losses)*100

print("\nTrain MNIST shape: ", x_train.shape)
print("Train MNIST anomaly detection rate: ", train_anomalies_percent, "%")


# ---------------------------------------------------------------------------- #
x_val_pred = autoencoder.predict(x_val)
validation_losses =[]
for i in range(len(x_val)):
      val_loss = mse(x_val[i], x_val_pred[i]).numpy()
      validation_losses.append(val_loss)

val_anomalies = [i for i, loss in enumerate(validation_losses) if loss>threshold]
val_anomalies_percent = len(val_anomalies)/len(validation_losses)*100

print("\nValidation MNIST shape: ", x_val.shape)
print("Validation MNIST anomaly detection rate: ", val_anomalies_percent, "%")

# ---------------------------------------------------------------------------- #
x_test_pred = autoencoder.predict(x_test)
test_losses =[]
for i in range(len(x_test)):
      test_loss = mse(x_test[i], x_test_pred[i]).numpy()
      test_losses.append(test_loss)

test_anomalies = [i for i, loss in enumerate(test_losses) if loss>threshold]
test_anomalies_percent = len(test_anomalies)/len(test_losses)*100

print("\nFashion MNIST shape: ", x_test.shape)
print("Fashion MNIST anomaly detection rate: ", test_anomalies_percent, "%")



exit()
