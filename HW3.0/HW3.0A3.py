import warnings
warnings.filterwarnings("ignore")

from keras.datasets import reuters
from keras.utils.np_utils import to_categorical

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers



import numpy as np
import matplotlib.pyplot as plt
# ---------------------------------------------- #
# ----------------- DISCLAIMER ----------------- #
# ---------------------------------------------- #

# I followed Chollet's notebook 
# https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/first_edition/3.6-classifying-newswires.ipynb

# and Prof Hickman's notebook
# https://github.com/jh2343/590-CODES/blob/main/LECTURE-CODES/WEEK5/C3-chollet-reuters-example.py

# ---------------------------------------------- #
# --------------- HYPERPARAMETERS -------------- #
# ---------------------------------------------- #
I_PLOT = True
I_RETRAIN = True

# ----------------- MODEL TYPE ----------------- #
# model_type = 'binary'
# output_activation = 'sigmoid'

model_type = 'multiclass'
output_activation = 'softmax'

# hidden_layer_size = 32
hidden_layer_size = 64
# hidden_layer_size = 128

num_epochs = 30
retrain_epochs = 8
my_batch_size = 512

f_train = 0.8           # Fraction for training vs validation set

# --------------- REGULARIZATION --------------- #
# GAMMA_L1 = 0.000; GAMMA_L2 = 0.000
# GAMMA_L1 = 0.001; GAMMA_L2 = 0.000
GAMMA_L1 = 0.000; GAMMA_L2 = 0.001
# GAMMA_L1 = 0.001; GAMMA_L2 = 0.001

# my_optimizer = 'rmsprop'
my_optimizer = 'Adam'
# my_optimizer = 'Nadam'
# my_optimizer = 'sgd'

# ----------------- REGRESSION ----------------- #
# my_loss_function = 'MeanSquaredError'
# my_loss_function = 'MeanAbsoluteError'
# my_loss_function = 'mean_squared_logarithmic_error'

# ------------ BINARY CLASSIFICATION ----------- #
# my_loss_function = 'binary_crossentropy'
# my_loss_function = 'hinge'
# my_loss_function = 'squared_hinge'

# ---------- MULTICLASS CLASSIFICATION --------- #
my_loss_function = 'categorical_crossentropy'
# my_loss_function = 'KLDivergence'

# ---------------------------------------------- #
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

def decode(data_point):
	word_index = reuters.get_word_index()
	reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
	decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in data_point])
	return decoded_newswire

print(decode(train_data[10]))

# ---------------------------------------------- #
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.  # set specific indices of results[i] to 1s
    return results

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# ---------------------------------------------- #

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# exit()

# ----------------- BUILD MODEL ---------------- #
input_shape=(x_train.shape[1],)

if model_type == 'binary':
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=input_shape,
    kernel_regularizer = regularizers.l1_l2(l1 = GAMMA_L1, l2 = GAMMA_L2)))
    model.add(layers.Dense(16, activation='relu',
    kernel_regularizer = regularizers.l1_l2(l1 = GAMMA_L1, l2 = GAMMA_L2)))
    model.add(layers.Dense(1, activation=output_activation))

if model_type == 'multiclass':
    model = models.Sequential()
    model.add(layers.Dense(hidden_layer_size, activation='relu', input_shape=input_shape,
    kernel_regularizer = regularizers.l1_l2(l1 = GAMMA_L1, l2 = GAMMA_L2)))
    model.add(layers.Dense(hidden_layer_size, activation='relu',
    kernel_regularizer = regularizers.l1_l2(l1 = GAMMA_L1, l2 = GAMMA_L2)))
    model.add(layers.Dense(46, activation=output_activation))

# ---------------- COMPILE MODEL --------------- #

model.compile(optimizer=my_optimizer,
              loss=my_loss_function,
              metrics=['accuracy'])

# ------------------ PATRITION ----------------- #

indices = np.random.permutation(x_train.shape[0])
CUT = int(f_train * x_train.shape[0])
partial_train_idx, val_idx = indices[:CUT], indices[CUT:]

x_val = x_train[val_idx]
partial_x_train = x_train[partial_train_idx]

y_val = one_hot_train_labels[val_idx]
partial_y_train = one_hot_train_labels[partial_train_idx]

print("partial train size: ", partial_x_train.shape[0], "val size: ", x_val.shape[0])
# exit()
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

    
def plot_loss():
    fig, ax = plt.subplots()
    # PLOTTING THE TRAINING AND VALIDATION LOSS 
    ax.plot(epochs, train_loss, "bo", label="Training loss")
    ax.plot(epochs, val_loss, "g", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
def plot_acc():
    # PLOTTING THE TRAINING AND VALIDATION ACCURACY
    fig, ax = plt.subplots()
    ax.plot(epochs, train_acc, "bo", label="Training accuracy")
    ax.plot(epochs, val_acc, "g", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# ---------------- INITIAL PLOTS --------------- #
if (I_PLOT):
    plot_loss()
    plot_acc()

# ------------------- RETRAIN ------------------ #

if (I_RETRAIN): 
    print("Retraining with ", retrain_epochs, " epochs")

    if model_type == 'binary':
        model = models.Sequential()
        model.add(layers.Dense(16, activation='relu', input_shape=input_shape,
        kernel_regularizer = regularizers.l1_l2(l1 = GAMMA_L1, l2 = GAMMA_L2)))
        model.add(layers.Dense(16, activation='relu',
        kernel_regularizer = regularizers.l1_l2(l1 = GAMMA_L1, l2 = GAMMA_L2)))
        model.add(layers.Dense(1, activation=output_activation))

    if model_type == 'multiclass':
        model = models.Sequential()
        model.add(layers.Dense(hidden_layer_size, activation='relu', input_shape=input_shape,
        kernel_regularizer = regularizers.l1_l2(l1 = GAMMA_L1, l2 = GAMMA_L2)))
        model.add(layers.Dense(hidden_layer_size, activation='relu',
        kernel_regularizer = regularizers.l1_l2(l1 = GAMMA_L1, l2 = GAMMA_L2)))
        model.add(layers.Dense(46, activation=output_activation))
    
    model.compile(optimizer=my_optimizer,
                loss=my_loss_function,
                metrics=['accuracy'])

    model.fit(partial_x_train,
            partial_y_train,
            epochs=retrain_epochs,
            batch_size=my_batch_size,
            validation_data=(x_val, y_val))

    print("Evaluating test set")

    results = model.evaluate(x_test, one_hot_test_labels)

