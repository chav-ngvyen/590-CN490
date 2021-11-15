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

n_bottleneck = 32
epochs = 100
batch_size = 1000
# Functional API representation of Prof Hickman's deep model in MNIST-DAE
# The code is borrowed from https://blog.keras.io/building-autoencoders-in-keras.html

# This is our input image
input_img = Input(shape=(28*28,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(256, activation='relu')(input_img)
encoded = layers.Dense(128, activation='relu')(encoded)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(n_bottleneck, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(256, activation='relu')(decoded)
decoded = layers.Dense(28*28, activation='sigmoid')(decoded)

# This model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# This model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# ---------------------------------------------------------------------------- #
mse = losses.MeanSquaredError()

autoencoder.compile(optimizer='adam',
                    loss= 'binary_crossentropy',
                    metrics = ['acc'])

print("\nAutoencoder summary:")
print(autoencoder.summary())
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
        plt.ylabel("Loss")
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
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_number[i].reshape(28, 28))
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)      
  
plt.savefig('./Plots/HW6.1_reconstruct_MNIST.png')
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

plt.savefig('./Plots/HW6.1_reconstruct_fashion_MNIST.png')
# plt.show()

# ---------------------------------------------------------------------------- #
# Define the anomaly threshold

x_train_pred = autoencoder.predict(x_train)


train_losses = []
for i in range(len(x_train)):
  train_loss = mse(x_train[i], x_train_pred[i]).numpy()
  train_losses.append(train_loss)

mean_loss = np.mean(train_losses)
print("MSE: ",mean_loss)
std_loss = np.std(train_losses)
print("Standard deviation: ", std_loss)
threshold = mean_loss+3*std_loss
print("Anomaly threshold: 3 std from the MSE: ", threshold)
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




# ---------------------------------------------------------------------------- #
# Plot of the cross entropies and anomaly threshold

fig, ax = plt.subplots()
ax.axhline(y=threshold, color='r', linestyle=':')
ax.plot(range(len(test_losses)), test_losses, color='lightgray',marker='.', alpha = 0.5, ms=1, label = "Fashion MNIST")
ax.plot(range(len(validation_losses)), validation_losses, color='k',marker='o', alpha = 0.75,  ms=1, label = "MNIST")
plt.title("Anomaly detection based on MSE")
plt.xlabel("Index")
plt.ylabel("MSE between original and reconstructed image")
plt.legend()
plt.savefig("./Plots/HW6.1_anomaly_detection.png")
# plt.show()