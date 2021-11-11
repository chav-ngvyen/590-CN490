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

# ---------------------------------------------------------------------------- #
# Sources: 
# A lot of this came directly from tf's tutorial on autoencoder
# https://www.tensorflow.org/tutorials/generative/
# this blog post
# https://medium.com/analytics-vidhya/image-anomaly-detection-using-autoencoders-ae937c7fd2d1
# Chapter 8 of Chollet
# and WEEK10 codes from Prof Hickman

# ---------------------------------------------------------------------------- #

# Training on MNIST dataset
# Get data

from tensorflow.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

#
# Normalize
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Reshape
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)

print("\n X train shape: ", x_train.shape)
print("\n X test shape: ", x_test.shape)

# ---------------------------------------------------------------------------- #
# Cholet
# seq_model = Sequential()
# seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
# seq_model.add(layers.Dense(32, activation='relu'))
# seq_model.add(layers.Dense(10, activation='softmax'))

# input_tensor = Input(shape=(64,))
# x = layers.Dense(32, activation='relu')(input_tensor)
# x = layers.Dense(32, activation='relu')(x)
# output_tensor = layers.Dense(10, activation='softmax')(x)

# Hickman
# model = models.Sequential()
# model.add(layers.Dense(400,  activation='relu', input_shape=(28 * 28,)))
# model.add(layers.Dense(n_bottleneck, activation='relu'))
# model.add(layers.Dense(400,  activation='relu'))
# model.add(layers.Dense(28*28,  activation='relu'))
n_bottleneck = 10

input_tensor = Input(shape = (28*28,))
encoded = layers.Dense(400, activation='relu')(input_tensor)
bottleneck = layers.Dense(n_bottleneck, activation='relu')(encoded)
decoded = layers.Dense(400, activation='relu')(bottleneck)
output_tensor = layers.Dense(28*28, activation = 'relu')(decoded)

autoencoder = Model(input_tensor, output_tensor)
autoencoder.summary()

autoencoder.compile(optimizer='rmsprop',
                    loss=losses.MeanSquaredError(),
                    metrics = ['acc'])

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))
print(dir(autoencoder))
exit()

latent_dim = config.N_BOTTLENECK

# Using the functional API

# Encoder
inputs = tf.keras.Input(shape=(28*28, ), name='input_layer')
# Flatten -> Dense
encoded = tf.keras.layers.Flatten()(inputs)
encoded = tf.keras.layers.Dense(latent_dim, activation = 'relu', name= 'bottleneck')(encoded)

# Decoder
# DeConv Block 1-> BatchNorm->leaky Relu
decoded = tf.keras.layers.Dense(28*28, activation = 'relu')(encoded)
decoded = tf.keras.layers.Reshape((28,28))(decoded)
outputs = tf.keras.layers.Dense(10,activation='sigmoid')(decoded)

# The Model class turns an input tensor and output tensor into a model
model = Model(inputs, outputs)
# Let's look at it!
model.summary()
exit()
# # ---------------------------------------------------------------------------- #
# # Define Autoecoder class
# # Latent dim is the number of nodes in the bottleneck
# latent_dim = config.N_BOTTLENECK

# class Autoencoder(Model):
#       def __init__(self, latent_dim):
#             super(Autoencoder, self).__init__()
#             self.latent_dim = latent_dim   
#             self.encoder = tf.keras.Sequential([
#               layers.Flatten(),
#               layers.Dense(latent_dim, activation='relu'),
#               ])
#             self.decoder = tf.keras.Sequential([
#               layers.Dense(28*28, activation='sigmoid'),
#               layers.Reshape((28, 28))
#               ])
#       def call(self, x):
#             encoded = self.encoder(x)
#             decoded = self.decoder(encoded)
#             return decoded

# autoencoder = Autoencoder(latent_dim)
# # ---------------------------------------------------------------------------- #
# autoencoder.compile(optimizer='rmsprop',
#                     loss=losses.MeanSquaredError(),
#                     metrics = ['acc'])
# print(x_train.shape)
# print(x_test.shape)
# # ---------------------------------------------------------------------------- #
# # Train the model using x_train as both the input and the target. 
# # The encoder will learn to compress the dataset from 784 dimensions to the latent space, 
# # and the decoder will learn to reconstruct the original images. .

# autoencoder.fit(x_train, x_train,
#                 epochs=config.EPOCHS,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))

# # ---------------------------------------------------------------------------- #
# # Now that the model is trained, let's test it by encoding and decoding images from the test set.

# encoded_imgs = autoencoder.encoder(x_test).numpy()
# decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

# # ---------------------------------------------------------------------------- #

# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#   # display original
#   ax = plt.subplot(2, n, i + 1)
#   plt.imshow(x_test[i])
#   plt.title("original")
#   plt.gray()
#   ax.get_xaxis().set_visible(False)
#   ax.get_yaxis().set_visible(False)

#   # display reconstruction
#   ax = plt.subplot(2, n, i + 1 + n)
#   plt.imshow(decoded_imgs[i])
#   plt.title("reconstructed")
#   plt.gray()
#   ax.get_xaxis().set_visible(False)
#   ax.get_yaxis().set_visible(False)
# plt.savefig('./Plots/HW6.1_autoencoder_reconstruct.png')
# plt.show()


# # ---------------------------------------------------------------------------- #
# # Define plotting functions

# def report(history,title='',I_PLOT=True):
#     if(I_PLOT):
#         #PLOT HISTORY
#         epochs = range(1, len(history.history['loss']) + 1)
#         plt.figure()
#         plt.plot(epochs, history.history['loss'], 'ro', label='Training loss')
#         plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
#         plt.title(title)
#         plt.legend()
#         plt.savefig('./Plots/HW6.1_autoencoder_history_loss.png')
#         plt.clf()
        
#         plt.plot(epochs, history.history['acc'], 'ro', label='Training acc')
#         plt.plot(epochs, history.history['val_acc'], 'b', label='Validation acc')
#         plt.title(title)
#         plt.legend()
#         plt.savefig('./Plots/HW6.1_autoencoder_history_acc.png')
#         plt.close()

# # Save the autoencoder plots
# report(autoencoder.history)

# # Save the autoencoder
# # Because autoencoder is a custom subclass, need to add save_format
# autoencoder.save('./Models/HW6.1_autoencoder', save_format='tf')

# x_train_pred = autoencoder.predict(x_train)
# print(x_train_pred.)