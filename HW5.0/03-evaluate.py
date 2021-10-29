import config
import numpy as np
import matplotlib.pyplot as plt
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_LOG_LEVEL

import tensorflow as tf
from tensorflow.keras.models import load_model
# ---------------------------------------------------------------------------- #
# Function to predict classes (undo softmax)
def predict_classes(input):
    ypred = model.predict(input)
    ypred_classes = np.argmax(ypred, axis =1 )
    return ypred_classes
# ---------------------------------------------------------------------------- #
# Load the trained model, print summary
model = load_model(config.best_model_path, compile = True)
print("\nModel summary: ", model.summary())

# ---------------------------------------------------------------------------- #
print("Applying model on test data, type: ", config.TEST)

x_train = np.load(config.processed_path+'x_train.npy'); print("X train shape: ", x_train.shape)
y_train = np.load(config.processed_path+'y_train.npy'); print("Y train shape: ", y_train.shape)
x_val = np.load(config.processed_path+'x_val.npy'); print("X val shape: ", x_val.shape)
y_val = np.load(config.processed_path+'y_val.npy'); print("Y val shape: ", y_val.shape)
x_test = np.load(config.processed_path+'x_test_'+config.TEST+'.npy'); print("X test shape: ", x_test.shape)
y_test = np.load(config.processed_path+'y_test_'+config.TEST+'.npy'); print("Y test shape: ", y_test.shape)


train_loss, train_acc = model.evaluate(x_train, y_train, batch_size=config.BATCH_SIZE)
val_loss, val_acc = model.evaluate(x_val, y_val, batch_size=config.BATCH_SIZE)
test_acc, test_loss = model.evaluate(x_test, y_test, batch_size = config.BATCH_SIZE)

print("MODEL PERFORMANCE")

print("\n Training accuracy: ", train_acc, "Training loss: ", train_loss)
print("\n Validation accuracy: ", val_acc, "Vlidation loss: ", val_loss)
print("\n Test accuracy: ", test_acc, "Test loss: ", test_loss)


# ---------------------------------------------------------------------------- #
# PLOTS
# Load training history for plotting
history=np.load(config.best_model_training_scores_path,allow_pickle='TRUE').item()
acc = history['acc']
val_acc = history['val_acc']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(1, len(acc) + 1)


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(config.plot_save_path+'_acc.png', dpi=300)
plt.show()
plt.clf()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(config.plot_save_path+'_loss.png', dpi=300)
plt.show()

exit()