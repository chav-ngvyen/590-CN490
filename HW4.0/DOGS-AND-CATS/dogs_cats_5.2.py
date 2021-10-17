import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# -------------------------------- SET PARAMS -------------------------------- #

# RESOURCE = "CPU"; import os; os.environ["CUDA_VISIBLE_DEVICES"]="-1"; import tensorflow as tf
RESOURCE = "CPU_GPU"; import os; os.environ["CUDA_VISIBLE_DEVICES"]="0"; import tensorflow as tf

if (RESOURCE == "CPU"):
        print("Using CPU")
        BATCH_SIZE = 200
        EPOCHS = 5
        STEPS_PER_EPOCH = 10
        VAL_STEPS = 1000/BATCH_SIZE

if (RESOURCE == "CPU_GPU"):
        print("Using CPU and GPU")
        BATCH_SIZE = 20
        EPOCHS = 30
        STEPS_PER_EPOCH = 100
        VAL_STEPS = 1000/BATCH_SIZE


''' NOTE: My current default is both because I am running Ubuntu 18.04 natively on a dual-boot Windows machine so Tensorflow has access to my NVIDIA GPU
If you are running Ubuntu on VM, Tensorflow can only see the CPU. Same with native macOS

Set CUDA VISIBLE DEVICES = 0 for both CPU and GPU
-1 for CPU ONLY

Source 1: https://stackoverflow.com/questions/44500733/tensorflow-allocating-gpu-memory-when-using-tf-device-cpu0/44518219#44518219
Source 2: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
'''

# if (RESOURCE == "CPU"):
#         os.environ["CUDA_VISIBLE_DEVICES"]="-1"; import tensorflow as tf
# if (RESOURCE == "CPU_GPU"):
#         os.environ["CUDA_VISIBLE_DEVICES"]="0"; import tensorflow as tf
# ---------------------------------------------------------------------------- #

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("Resource: ", RESOURCE, "batch size: ", BATCH_SIZE, "N epochs: ", EPOCHS, "Steps per epoch: ", STEPS_PER_EPOCH)

# ---------------------------------------------- #
#                    Set paths                   #
# ---------------------------------------------- #

# Read in the paths created in setup_dogs_cats.py
base_dir = '/home/chau/590-CN490/HW4.0/DOGS-AND-CATS/Data_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test1')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

# ---------------------------------------------- #
#                       5.2                      #
# ---------------------------------------------- #

# Initiate model
model = models.Sequential()
# Convolutional 
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
# Dense feed forward
model.add(layers.Dense(512, activation='relu'))
# Output layer, binary classification so use sigmoid as activation function
model.add(layers.Dense(1, activation='sigmoid'))

# Print out dimension
model.summary()
# exit()

# ---------------------------------------------- #
# Use binary crossentropy because binary classification
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# ---------------------------------------------- #
# Rescale pixel values from [0,255] to [0,1]
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        #Increased batch size for faster run
        batch_size=BATCH_SIZE,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=BATCH_SIZE,
        class_mode='binary')

print()
# Need to break loop because this is a generator class
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit(
      train_generator,
      steps_per_epoch=STEPS_PER_EPOCH,
      epochs=EPOCHS,
      validation_data=validation_generator,
      validation_steps=VAL_STEPS)

# Save the model
model.save('./Models/5.2/cats_and_dogs_small_1.h5')
# exit()
# ---------------------------------------------- #
# Plot
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc', color = 'blue')
plt.plot(epochs, val_acc, 'b', label='Validation acc', color = 'blue')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss', color = 'orange')
plt.plot(epochs, val_loss, 'b', label='Validation loss', color = 'orange')
plt.title('Training and validation loss')
plt.legend()
# Non-blocking plot
plt.show(block = False)

# ---------------------------------------------- #

# ---------------------------------------------- #
#                Data augmentation               #
# ---------------------------------------------- #


datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

# We pick one image to "augment"
img_path = fnames[3]

# Read the image and resize it
img = image.load_img(img_path, target_size=(150, 150))

# Convert it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)

# Reshape it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

# ---------------------------------------------- #

# Initiate another model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Add augmentation to the data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# ---------------------------------------------- #
# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=BATCH_SIZE,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=BATCH_SIZE,
        class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=STEPS_PER_EPOCH,
      epochs=EPOCHS,
      validation_data=validation_generator,
      validation_steps=VAL_STEPS)
# Save model
model.save('./Models/5.2/cats_and_dogs_small_2.h5')

# ---------------------------------------------- #
# Plot
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc', color = 'blue')
plt.plot(epochs, val_acc, 'b', label='Validation acc', color = 'blue')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss', color = 'orange')
plt.plot(epochs, val_loss, 'b', label='Validation loss', color = 'orange')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Don't include block=False here so that the script ends
plt.show()
