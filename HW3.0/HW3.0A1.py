import warnings
warnings.filterwarnings("ignore")

from keras.datasets import boston_housing
# from keras.utils.np_utils import to_categorical

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
# https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/first_edition/3.7-predicting-house-prices.ipynb
# and Prof Hickman's notebook
# https://github.com/jh2343/590-CODES/blob/main/LECTURE-CODES/WEEK3/D2.2-MPG-KERAS-REGRESSION.py

# ---------------------------------------------- #
# --------------- HYPERPARAMETERS -------------- #
# ---------------------------------------------- #
I_PLOT = True
I_RETRAIN = False
I_NORMALIZE = True

# ----------------- MODEL TYPE ----------------- #
# model_type = 'logistic'

# model_type = 'linear'

model_type = 'ann'

output_activation = 'linear'

input_activation = 'sigmoid'
# ---------------------------------------------- #

# hidden_layer_size = 32
hidden_layer_size = 64
# hidden_layer_size = 128

num_epochs = 20
retrain_epochs = 8
my_batch_size = 1

f_train = 0.8           # Fraction for training vs validation set

k = 4                   # K-FOLDS


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
my_loss_function = 'MeanSquaredError'
# my_loss_function = 'MeanAbsoluteError'
# my_loss_function = 'mean_squared_logarithmic_error'

# ------------ BINARY CLASSIFICATION ----------- #
# my_loss_function = 'binary_crossentropy'
# my_loss_function = 'hinge'
# my_loss_function = 'squared_hinge'

# ---------- MULTICLASS CLASSIFICATION --------- #
# my_loss_function = 'categorical_crossentropy'
# my_loss_function = 'KLDivergence'

# ------------------ READ DATA ----------------- #
(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()


x_mean=np.mean(train_data,axis=0); x_std=np.std(train_data,axis=0)
y_mean=np.mean(train_targets,axis=0); y_std=np.std(train_targets,axis=0)


if(I_NORMALIZE):
    train_data=(train_data-x_mean)/x_std
    test_data=(test_data-x_mean)/x_std
    
    train_targets=(train_targets-y_mean)/y_std
    test_targets=(test_targets-y_mean)/y_std

    I_UNNORMALIZE = True
else:
    I_UNNORMALIZE = False

# -------------- DEFINE FUNCTIONS -------------- #
def build_model():
    if model_type == 'logistic':
        model = models.Sequential([
            layers.Dense(1,
            activation = output_activation,
            input_shape = (train_data.shape[1],))
        ])

    if model_type == 'linear':
        model = models.Sequential([
            layers.Dense(1,
            activation = output_activation,
            input_shape = (train_data.shape[1],))
        ])

    if model_type == 'ann':
        model = models.Sequential()
        model.add(layers.Dense(hidden_layer_size,activation=input_activation,
                               input_shape=(train_data.shape[1],)))
        model.add(layers.Dense(hidden_layer_size,activation=input_activation))      
        model.add(layers.Dense(1,activation = output_activation))
    
    model.compile(optimizer=my_optimizer, loss=my_loss_function, metrics=['mae'])
    return model

# ---------------------------------------------- #

num_val_samples = len(train_data) // k

train_mae_histories = []
val_mae_histories = []
train_loss_histories = []
val_loss_histories = []

for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=my_batch_size, verbose=1)
    
    history_dict = history.history
 
    train_mae = history_dict["mae"]
    train_mae_histories.append(train_mae)

    val_mae = history_dict["val_mae"]
    val_mae_histories.append(val_mae)

    train_loss = history_dict["loss"]
    train_loss_histories.append(train_loss)
    
    val_loss = history_dict["val_loss"]
    val_loss_histories.append(val_loss)

# ---------------------------------------------- #


avg_train_mae_history = [
    np.mean([x[i] for x in train_mae_histories]) for i in range(num_epochs)]

avg_train_loss_history = [
    np.mean([x[i] for x in train_loss_histories]) for i in range(num_epochs)]

avg_val_mae_history = [
    np.mean([x[i] for x in val_mae_histories]) for i in range(num_epochs)]

avg_val_loss_history = [
    np.mean([x[i] for x in val_loss_histories]) for i in range(num_epochs)]

# ---------------------------------------------- #
# plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()

plt.plot(range(1, len(avg_train_loss_history) + 1), avg_train_loss_history, "bo", label="Training loss")
plt.plot(range(1, len(avg_val_loss_history) + 1), avg_val_loss_history, "rx", label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Errors')
plt.legend()
plt.show()

plt.plot(range(1, len(avg_train_mae_history) + 1), avg_train_mae_history, "bo", label="Training loss")
plt.plot(range(1, len(avg_val_mae_history) + 1), avg_val_mae_history, "rx", label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Errors')
plt.legend()
plt.show()
    
yp=model.predict(train_data)
yp_test=model.predict(test_data) 

if I_UNNORMALIZE:

#UN-NORMALIZE DATA (CONVERT BACK TO ORIGINAL UNITS)
    x_train=x_std*train_data+x_mean 
    y_train=y_std*train_targets+y_mean 
    yp=y_std*yp+y_mean 
    yp_test=y_std*yp_test+y_mean  

plt.plot(yp_test  , test_targets,'*') 
# plt.plot(Y[val_idx]    , YPRED_V,'*', label='Validation') 
# plt.xlabel(xla, fontsize=18);	plt.ylabel(yla, fontsize=18); 	plt.legend()
plt.xlabel("y (predicted)")
plt.ylabel("y (data)")
# plt.legend()
plt.show()
# def plot_loss():
#     fig, ax = plt.subplots()
#     # PLOTTING THE TRAINING AND VALIDATION LOSS 
#     ax.plot(epochs, train_loss, "bo", label="Training loss")
#     ax.plot(epochs, val_loss, "g", label="Validation loss")
#     plt.title("Training and validation loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.show()
    
# def plot_acc():
#     # PLOTTING THE TRAINING AND VALIDATION ACCURACY
#     fig, ax = plt.subplots()
#     ax.plot(epochs, train_acc, "bo", label="Training accuracy")
#     ax.plot(epochs, val_acc, "g", label="Validation accuracy")
#     plt.title("Training and validation accuracy")
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.show()

# # ---------------- INITIAL PLOTS --------------- #
# if (I_PLOT):
#     plot_loss()
#     plot_acc()