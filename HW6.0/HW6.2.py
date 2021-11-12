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

n_bottleneck = 100
epochs = 1
batch_size = 5000
# Functional API representation of Prof Hickman's deep model in MNIST-DAE
# The code is borrowed from https://blog.keras.io/building-autoencoders-in-keras.html

input_img = Input(shape=(28, 28, 1))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
#x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = layers.MaxPooling2D((2, 2), padding='same')(x)
encoded = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# "decoded" is the loss reconstruction of the input
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
#x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# # This model maps an input to its reconstruction
# autoencoder = Model(input_img, decoded)
# # This model maps an input to its encoded representation
# encoder = Model(input_img, encoded)

# # ---------------------------------------------------------------------------- #
# autoencoder.compile(optimizer='adam',
#                     loss='binary_crossentropy',
#                     metrics = ['acc'])

# print("\nAutoencoder summary:")
# print(autoencoder.summary())


# # Create the Encoder and Decoder#pass the gray scale input image of size(28,28,1)
# inputs = Input(shape=(28, 28, 1), name='input_layer')
# # Conv Block 1 -> BatchNorm->leaky Relu
# encoded = layers.Conv2D(32, kernel_size=3, strides= 1, padding='same', name='conv_1')(inputs)
# encoded = layers.BatchNormalization(name='batchnorm_1')(encoded)
# encoded = layers.LeakyReLU(name='leaky_relu_1')(encoded)# Conv Block 2 -> BatchNorm->leaky Relu
# encoded = layers.Conv2D(64, kernel_size=3, strides= 2, padding='same', name='conv_2')(encoded)
# encoded = layers.BatchNormalization(name='batchnorm_2')(encoded)
# encoded = layers.LeakyReLU(name='leaky_relu_2')(encoded)
# # Conv BloatchNorm->leaky Relu
# encoded = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', name='conv_3')(encoded)
# encoded = layers.BatchNormalization(name='batchnorm_3')(encoded)
# encoded = layers.LeakyReLU(name='leaky_relu_3')(encoded)#Decoder
# # DeConv BBatchNorm->leaky Relu
# decoded = layers.Conv2DTranspose(64, 3, strides= 1, padding='same',name='conv_transpose_1')(encoded)
# decoded = layers.BatchNormalization(name='batchnorm_4')(decoded)
# decoded = layers.LeakyReLU(name='leaky_relu_4')(decoded)
# # DeConv BBatchNorm->leaky Relu
# decoded = layers.Conv2DTranspose(64, 3, strides= 2, padding='same', name='conv_transpose_2')(decoded)
# decoded = layers.BatchNormalization(name='batchnorm_5')(decoded)
# decoded = layers.LeakyReLU(name='leaky_relu_5')(decoded)
# # DeConv Block 3-> BatchNorm->leaky Relu
# decoded = layers.Conv2DTranspose(32, 3, 2, padding='same', name='conv_transpose_3')(decoded)
# decoded = layers.BatchNormalization(name='batchnorm_6')(decoded)
# decoded = layers.LeakyReLU(name='leaky_relu_6')(decoded)
# # output
# outputs = layers.Conv2DTranspose(1, 3, 1,padding='same', activation='sigmoid', name='conv_transpose_4')(decoded)

# # ---------------------------------------------------------------------------- #
def SSIMLoss(y_true, y_pred):
      return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,1.0))


autoencoder = tf.keras.Model(input_img, decoded)
# optimizer = tf.keras.optimizers.Adam(lr = 0.0005)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['acc'])

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
        plt.ylabel("Binary Crossentropy")
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

#encoded_fashion = encoder.predict(x_test)
decoded_fashion = autoencoder.predict(x_test)

#encoded_number = encoder.predict(x_val)
decoded_number = autoencoder.predict(x_val)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_val[i].reshape(28, 28, 1))
  plt.title("original number")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_number[i].reshape(28, 28,1))
  plt.title("reconstructed number")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)      
  
plt.savefig('./Plots/HW6.2_reconstruct_number.png')
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

plt.savefig('./Plots/HW6.2_reconstruct_fashion.png')
# plt.show()


# ---------------------------------------------------------------------------- #
# Define the anomaly threshold

x_train_pred = autoencoder.predict(x_train)


mse = losses.MeanSquaredError()

losses = mse(x_train, x_train_pred).numpy()


mean_loss = np.mean(losses)
print("Mean binary_crossentropy: ",mean_loss)
std_loss = np.std(losses)
print("Standard deviation: ", std_loss)
threshold = mean_loss+3*std_loss
print("Anomaly threshold: 3 std from the mean binary crossentropy: ", threshold)

exit()

# ---------------------------------------------------------------------------- #
# # Find the anomaly rate for the normal MNIST validation
# train_reconstruction_loss = np.asarray(reconstruction_loss)


# train_anomalies = [i for i, loss in enumerate(train_reconstruction_loss[0]) if loss>threshold]
# print(train_anomalies.shape)
# train_anomalies_percent = len(train_anomalies)/len(train_reconstruction_loss)*100

# print("\nTrain MNIST shape: ", x_train.shape)
# print("Train MNIST anomaly detection rate: ", train_anomalies_percent, "%")

# ---------------------------------------------------------------------------- #
# https://www.analyticsvidhya.com/blog/2021/05/anomaly-detection-using-autoencoders-a-walk-through-in-python/
# This blog has the 
def find_threshold(model, x_true):
    x_pred = model.predict(x_true)
    # provides losses of individual instances
    reconstruction_loss = losses.binary_crossentropy(x_true, x_pred)
    # threshold for anomaly scores
    threshold = np.mean(reconstruction_loss.numpy()) + 3*np.std(reconstruction_loss.numpy())
    return threshold

# print(find_threshold(autoencoder,x_train))


def get_predictions(model, x_test, threshold):
    x_test_pred = model.predict(x_test)
    # provides losses of individual instances
    errors = losses.binary_crossentropy(x_test, x_test_pred).numpy()
    # 0 = anomaly, 1 = normal
    anomaly_mask = pd.Series(errors) > threshold
    preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
    return preds

threshold = find_threshold(autoencoder,x_train)
print(f"Threshold: {threshold}")
# Threshold: 0.01001314025746261
x_val_pred = get_predictions(autoencoder, x_val, threshold)
accuracy_score(x_val_pred, x_val)
# 0.944

exit()
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
plt.savefig("./Plots/HW6.2_cross_entropy_detection.png")
# plt.show()