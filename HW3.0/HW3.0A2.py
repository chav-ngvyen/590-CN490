import warnings
warnings.filterwarnings("ignore")

from keras.datasets import imdb

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers

import numpy as np
import matplotlib.pyplot as plt
# ---------------------------------------------- #
# ----------------- DISCLAIMER ----------------- #
# ---------------------------------------------- #

# I followed Chollet's notebook https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/first_edition/3.5-classifying-movie-reviews.ipynb 
# and D2.2-IMDB-LOGISTIC.py in Week3 slides

# ---------------------------------------------- #
# --------------- HYPERPARAMETERS -------------- #
# ---------------------------------------------- #
I_PLOT = True

model_type = 'binary'

num_epochs = 30
retrain_epochs = 5
my_batch_size = 10000

GAMMA_L1 = 0.000
GAMMA_L2 = 0.001

# ---------------------------------------------- #
my_optimizer = 'rmsprop'

# ----------------- REGRESSION ----------------- #
# my_loss_function = 'MeanSquaredError'
# my_loss_function = 'MeanAbsoluteError'
# my_loss_function = 'mean_squared_logarithmic_error'

# ------------ BINARY CLASSIFICATION ----------- #
my_loss_function = 'binary_crossentropy'
# my_loss_function = 'hinge'
# my_loss_function = 'squared_hinge'


# ---------------------------------------------- #
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.  # set specific indices of results[i] to 1s
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#COMPARE OLD VS NEW
indx = 0
print(x_train[indx].shape)
print(train_data[indx])

print(sorted(train_data[indx]))
print(x_train[indx][0:30])
print(y_train[0:30])

# exit()

# ----------------- BUILD MODEL ---------------- #
input_shape=(x_train.shape[1],)

if model_type == 'binary':
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=input_shape,
    kernel_regularizer = regularizers.l1_l2(l1 = GAMMA_L1, l2 = GAMMA_L2)))
    model.add(layers.Dense(16, activation='relu',
    kernel_regularizer = regularizers.l1_l2(l1 = GAMMA_L1, l2 = GAMMA_L2)))
    model.add(layers.Dense(1, activation='sigmoid'))

# ---------------- COMPILE MODEL --------------- #

model.compile(optimizer=my_optimizer,
              loss=my_loss_function,
              metrics=['accuracy'])

# -------------------- SPLIT ------------------- #


x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# -------------------- TRAIN ------------------- #

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=num_epochs,
                    batch_size=my_batch_size,
                    validation_data=(x_val, y_val))

# ------------------- PREDICT ------------------ #

# yp=model.predict(x_train)
# yp_val=model.predict(x_val) 

# ----------- SAVE DATA FOR PLOTTING ----------- #
history_dict = history.history

train_loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(train_loss) + 1)
# -------------------- PLOT -------------------- #

#PLOT INITIAL DATA
if(I_PLOT):
    FS=18   #FONT SIZE

    # plt.plot(x_train, y_train, 'o', label = 'Training set')
    # plt.plot(x_val, y_val, 'rx', label = 'Validation set')
    # plt.plot(x_test, y_test, 'bx', label = 'Test set')
    # plt.xlabel('x')
    # plt.yclabel('y')
    # plt.legend()
    # plt.show()
    # plt.clf()

    # #PARITY PLOT
    # plt.plot(yp,yp,'-')
    # plt.plot(yp,y_train,'o')
    # plt.xlabel("y (predicted)", fontsize=FS)
    # plt.ylabel("y (data)", fontsize=FS)
    # plt.show()
    # plt.clf()

    # #FEATURE DEPENDENCE
    # for indx in range(0,x_train.shape[1]):
    #     #TRAINING
    #     plt.plot(x_train[:,indx],y_train,'ro')
    #     plt.plot(x_train[:,indx],yp,'bx')
    #     plt.xlabel(x_keys[indx], fontsize=FS)
    #     plt.ylabel(y_keys[0], fontsize=FS)
    #     plt.show()
    #     plt.clf()

    #     plt.plot(x_val[:,indx],y_val,'ro')
    #     plt.plot(x_val[:,indx],yp_val,'bx')
    #     plt.xlabel(x_keys[indx], fontsize=FS)
    #     plt.ylabel(y_keys[0], fontsize=FS)
    #     plt.show()
    #     plt.clf()

    

    # PLOTTING THE TRAINING AND VALIDATION LOSS 
    plt.plot(epochs, train_loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.clf()

    # PLOTTING THE TRAINING AND VALIDATION ACCURACY 
    plt.plot(epochs, train_acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# ----------- RETRAIN WITH EPOCHS = 5 ---------- #

print("Retraining with ", retrain_epochs, " epochs")

if model_type == 'binary':
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=input_shape,
    kernel_regularizer = regularizers.l1_l2(l1 = GAMMA_L1, l2 = GAMMA_L2)))
    model.add(layers.Dense(16, activation='relu',
    kernel_regularizer = regularizers.l1_l2(l1 = GAMMA_L1, l2 = GAMMA_L2)))
    model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer=my_optimizer,
              loss=my_loss_function,
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=retrain_epochs,
          batch_size=my_batch_size,
          validation_data=(x_val, y_val))

print("Evaluating test set")

results = model.evaluate(x_test, y_test)

