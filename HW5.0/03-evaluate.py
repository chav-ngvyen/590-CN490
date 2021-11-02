import config
import numpy as np
import matplotlib.pyplot as plt
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_LOG_LEVEL

import tensorflow as tf
from tensorflow.keras.models import load_model
from config import MODEL_LIST
# ---------------------------------------------------------------------------- #
# Function to predict classes (undo softmax)
def predict_classes(input):
    ypred = model.predict(input)
    ypred_classes = np.argmax(ypred, axis =1 )
    return ypred_classes
# ---------------------------------------------------------------------------- #
# Load the training & test data
print("Loading data")
x_train = np.load(config.processed_path+'x_train.npy'); print("X train shape: ", x_train.shape)
y_train = np.load(config.processed_path+'y_train.npy'); print("Y train shape: ", y_train.shape)
x_val = np.load(config.processed_path+'x_val.npy'); print("X val shape: ", x_val.shape)
y_val = np.load(config.processed_path+'y_val.npy'); print("Y val shape: ", y_val.shape)

print("\nTest CORPUS means that the test set came from the same books as training & validation")
x_test_CORPUS = np.load(config.processed_path+'x_test_CORPUS.npy'); print("X test (CORPUS) shape: ", x_test_CORPUS.shape)
y_test_CORPUS = np.load(config.processed_path+'y_test_CORPUS.npy'); print("Y test (CORPUS) shape: ", y_test_CORPUS.shape)

print("\nTest UNIVERSE means that the test set came from the same universe/ series as training & validation")
x_test_UNIVERSE = np.load(config.processed_path+'x_test_UNIVERSE.npy'); print("X test (UNIVERSE) shape: ", x_test_UNIVERSE.shape)
y_test_UNIVERSE = np.load(config.processed_path+'y_test_UNIVERSE.npy'); print("Y test (UNIVERSE) shape: ", y_test_UNIVERSE.shape)

print("\nTest AUTHOR means that the test set came from writings about topics/ characters outside of training & validation")
x_test_AUTHOR = np.load(config.processed_path+'x_test_AUTHOR.npy'); print("X test (AUTHOR) shape: ", x_test_AUTHOR.shape)
y_test_AUTHOR = np.load(config.processed_path+'y_test_AUTHOR.npy'); print("Y test (AUTHOR) shape: ", y_test_AUTHOR.shape)

# ---------------------------------------------------------------------------- #
# Load the trained models in path, print summary
model_path = os.path.join("./Models")
for mname in os.listdir("./Models"):
    if mname[-5:]==".hdf5":
        print("-------------------------")
        print("LOADING: ", mname)
        print("-------------------------")
        
        model = load_model("./Models/"+mname, compile = True)
        print("\nFITTING MODELS: ")

        print("Training set: ")
        train_loss, train_acc = model.evaluate(x_train, y_train, batch_size=config.BATCH_SIZE)
        print("Validation set: ")
        val_loss, val_acc = model.evaluate(x_val, y_val, batch_size=config.BATCH_SIZE)
        print("Test CORPUS: ")
        test_loss_CORPUS, test_acc_CORPUS = model.evaluate(x_test_CORPUS, y_test_CORPUS, batch_size = config.BATCH_SIZE)
        print("Test UNIVERSE: ")
        test_loss_UNIVERSE, test_acc_UNIVERSE = model.evaluate(x_test_UNIVERSE, y_test_UNIVERSE, batch_size = config.BATCH_SIZE)
        print("Test AUTHOR: ")
        test_loss_AUTHOR, test_acc_AUTHOR = model.evaluate(x_test_AUTHOR, y_test_AUTHOR, batch_size = config.BATCH_SIZE)

        print("----------------------")
        print("MODEL PERFORMANCE")
        print("----------------------")

        print("\n Training accuracy: ", train_acc, "Training loss: ", train_loss)
        print("\n Validation accuracy: ", val_acc, "Validation loss: ", val_loss)
        
        print("--------------------")
        print("\nTest CORPUS means that the test set came from the same books as training & validation")
        print("\n Test accuracy (CORPUS): ", test_acc_CORPUS, "Test loss (CORPUS): ", test_loss_CORPUS)
        
        print("--------------------")
        print("\nTest UNIVERSE means that the test set came from the same universe/ series as training & validation")        
        print("\n Test accuracy (UNIVERSE): ", test_acc_UNIVERSE, "Test loss (UNIVERSE): ", test_loss_UNIVERSE)
        
        print("--------------------")
        print("\nTest AUTHOR means that the test set came from writings about topics/ characters outside of training & validation")
        print("\n Test accuracy (AUTHOR): ", test_acc_AUTHOR, "Test loss (AUTHOR): ", test_loss_AUTHOR)

# # ---------------------------------------------------------------------------- #

# exit()