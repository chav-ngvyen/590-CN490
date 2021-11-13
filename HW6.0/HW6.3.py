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

from tensorflow.keras.datasets import cifar10, cifar100

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
epochs = 50
batch_size = 10000

# This came from TF's tutorial
# https://www.tensorflow.org/tutorials/images/cnn


input_img = Input(shape=(32, 32, 3))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

# "decoded" is the loss reconstruction of the input
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['acc'])

autoencoder.summary()
# exit()
# ---------------------------------------------------------------------------- #
(x_train, y_train), (x_val, y_val) = cifar10.load_data()
# Remove pickup truck from CIFAR10

print("\nRemoving pickup truck from CIFAR 10")

x_train = x_train[y_train.flatten() != 9]
x_val = x_val[y_val.flatten() != 9]


# Normalize data
x_train = x_train.reshape((x_train.shape[0],) + (32, 32, 3)).astype('float32') / 255.
x_val = x_val.reshape((x_val.shape[0],) + (32, 32, 3)).astype('float32') / 255.

print("CIFAR10 shape: ", x_train.shape)
print("CIFAR10 test shape: ", x_val.shape)
# exit()
# exit()
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
        plt.savefig('./Plots/HW6.3_autoencoder_history_loss.png')
        plt.clf()
        
        plt.plot(epochs, history.history['acc'], 'ro', label='Training acc')
        plt.plot(epochs, history.history['val_acc'], 'b-', label='Validation acc')
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('./Plots/HW6.3_autoencoder_history_acc.png')
        plt.close()

# Save the autoencoder plots
report(history)
# Save the autoencoder
autoencoder.save('./Models/HW6.3_autoencoder.hdf5')

print("Finished training")


# ---------------------------------------------------------------------------- #
# Read in the CIFAR-100 data
print("Reading in CIFAR100")
print("Not removing pickup truck")
(_, _), (x_test, y_test) = cifar100.load_data()
# Remove pickup truck from CIFAR100
# print("\nRemoving pickup truck from CIFAR 100")

# x_test = x_test[y_test.flatten() != 58]

# Prepare the data
x_test = x_test.reshape((x_test.shape[0],) + (32, 32, 3)).astype('float32') / 255.

print("\nCIFAR-100 shape: ", x_test.shape)



# ---------------------------------------------------------------------------- #
# Prints out how the autoencoder does

decoded_cifar100 = autoencoder.predict(x_test)
decoded_cifar10 = autoencoder.predict(x_val)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_val[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_cifar10[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)      
  
plt.savefig('./Plots/HW6.3_reconstruct_CIFAR10.png')


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_cifar100[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.savefig('./Plots/HW6.3_reconstruct_cifar100.png')
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
print("Mean MSE: ",mean_loss)
std_loss = np.std(train_losses)
print("Standard deviation: ", std_loss)
threshold = mean_loss+3*std_loss
print("Anomaly threshold: 3 std from the mean: ", threshold)
max_loss = np.max(train_loss)
print("\nMax loss: ", max_loss)

train_anomalies = [i for i, loss in enumerate(train_losses) if loss>threshold]
train_anomalies_percent = len(train_anomalies)/len(train_losses)*100

print("\nTrain CIFAR10 shape: ", x_train.shape)
print("Train CIFAR10 anomaly detection rate: ", train_anomalies_percent, "%")


# ---------------------------------------------------------------------------- #
x_val_pred = autoencoder.predict(x_val)
validation_losses =[]
for i in range(len(x_val)):
      val_loss = mse(x_val[i], x_val_pred[i]).numpy()
      validation_losses.append(val_loss)

val_anomalies = [i for i, loss in enumerate(validation_losses) if loss>threshold]
val_anomalies_percent = len(val_anomalies)/len(validation_losses)*100

print("\nValidation CIFAR10 shape: ", x_val.shape)
print("Validation CIFAR10 anomaly detection rate: ", val_anomalies_percent, "%")

# ---------------------------------------------------------------------------- #
x_test_pred = autoencoder.predict(x_test)
test_losses =[]
for i in range(len(x_test)):
      test_loss = mse(x_test[i], x_test_pred[i]).numpy()
      test_losses.append(test_loss)

test_anomalies = [i for i, loss in enumerate(test_losses) if loss>threshold]
test_anomalies_percent = len(test_anomalies)/len(test_losses)*100

print("\nTest CIFAR100 shape: ", x_test.shape)
print("Test CIFAR100 anomaly detection rate: ", test_anomalies_percent, "%")

# ---------------------------------------------------------------------------- #
# Read in the CIFAR-100 data
print("Reading in CIFAR100 again")
(x_test, y_test), (_, _) = cifar100.load_data()
# indices = np.arange(x_test.shape[0])
# np.random.shuffle(indices)

# x_test = x_test[indices]
# y_test = y_test[indices]


# print(x_test.shape); exit()
print("\nSplit into 2, remove pickup truck from 1 half and leave it in in the other half")
x_test_with_truck = x_test[0:25000]; y_test_with_truck = y_test[0:25000]
x_test_without_truck = x_test[25000:]; y_test_without_truck = y_test[25000:]


# Remove pickup truck from CIFAR100
print("\nRemoving pickup truck from CIFAR 100")

x_test_without_truck = x_test_without_truck[y_test_without_truck.flatten() != 58]

# Prepare the data
x_test_with_truck = x_test_with_truck.reshape((x_test_with_truck.shape[0],) + (32, 32, 3)).astype('float32') / 255.
x_test_without_truck = x_test_without_truck.reshape((x_test_without_truck.shape[0],) + (32, 32, 3)).astype('float32') / 255.


# ---------------------------------------------------------------------------- #
x_test_with_truck_pred = autoencoder.predict(x_test_with_truck)
test_with_truck_losses =[]

for i in range(len(x_test_without_truck)):
      test_with_truck_loss = mse(x_test_with_truck[i], x_test_with_truck_pred[i]).numpy()
      test_with_truck_losses.append(test_with_truck_loss)

test_with_truck_anomalies = [i for i, loss in enumerate(test_with_truck_losses) if loss>threshold]
test_with_truck_anomalies_percent = len(test_with_truck_anomalies)/len(test_with_truck_losses)*100

print("\nWith truck shape : ", x_test_with_truck.shape)
print("\nWith truck anomaly detection rate: ", test_with_truck_anomalies_percent, "%")



# ---------------------------------------------------------------------------- #
x_test_without_truck_pred = autoencoder.predict(x_test_without_truck)
test_without_truck_losses =[]

for i in range(len(x_test_without_truck)):
      test_without_truck_loss = mse(x_test_without_truck[i], x_test_without_truck_pred[i]).numpy()
      test_without_truck_losses.append(test_without_truck_loss)

test_without_truck_anomalies = [i for i, loss in enumerate(test_without_truck_losses) if loss>threshold]
test_without_truck_anomalies_percent = len(test_without_truck_anomalies)/len(test_without_truck_losses)*100

print("\nWithout truck shape : ", x_test_without_truck.shape)
print("\nWithout truck anomaly detection rate: ", test_without_truck_anomalies_percent, "%")


exit()
