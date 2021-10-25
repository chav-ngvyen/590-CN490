import config
import numpy as np
from tensorflow.keras.models import load_model

to_load = config.model_save_path+".h5"
# print(to_load); exit()
model = load_model(to_load, compile = True)
print(model.summary())
exit()
print("Applying model on test data")
x_test = np.load("./Data/x_test.npy"); print("X test shape: ", x_test.shape)
y_test = np.load("./Data/y_test.npy"); print("Y test shape: ", y_test.shape)

model.evaluate(x_test, y_test)
