import config
import numpy as np
import matplotlib.pyplot as plt
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_LOG_LEVEL

import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model(config.best_model_path, compile = True)
print("\nModel summary: ", model.summary())

history=np.load(config.best_model_training_scores_path,allow_pickle='TRUE').item()

# ---------------------------------------------------------------------------- #
print("Applying model on test data, type: ", config.TEST)
x_test = np.load(config.processed_path+'x_test_'+config.TEST+'.npy'); print("X test shape: ", x_test.shape)
y_test = np.load(config.processed_path+'y_test_'+config.TEST+'.npy'); print("Y test shape: ", y_test.shape)

test_acc, test_loss = model.evaluate(x_test, y_test)
print("TEST:",test_acc, test_loss)

# ---------------------------------------------------------------------------- #

acc = history['acc']
val_acc = history['val_acc']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.plot(epochs,test_acc, 'rx', label = 'Test accuracy' )
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.savefig(config.plot_save_path+'_acc.png', dpi=300)
# plt.draw()
plt.show(block=False)
#plt.clf()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs,test_loss, 'rx', label = 'Test loss' )
plt.title('Training and validation loss')
plt.legend()
plt.savefig(config.plot_save_path+'_loss.png', dpi=300)
plt.show()

exit()