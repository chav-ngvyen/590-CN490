import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
# ---------------------------------------------------------------------------- #
#                                  PARAMETERS                                  #
# ---------------------------------------------------------------------------- #

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
        EPOCHS = 20
        STEPS_PER_EPOCH = 10
        VAL_STEPS = 1000/BATCH_SIZE

# ---------------------------------------------------------------------------- #

# RUN_FEATURE_EXTRACT_PRETRAINED = 1
RUN_FEATURE_EXTRACT_PRETRAINED = 0

# RUN_FEATURE_EXTRACT_DATA_AUG = 1
RUN_FEATURE_EXTRACT_DATA_AUG = 0

RUN_FINE_TUNING = 1
# RUN_FINE_TUNING = 0

# ---------------------------------------------------------------------------- #
#                   FEATURE EXTRACTION USING PRETRAINED MODEL                  #
# ---------------------------------------------------------------------------- #


# Instantiate pre-trained VGG16 model
conv_base = VGG16(weights = 'imagenet',
            include_top = False,
            input_shape = (150, 150, 3))

conv_base.summary()

# Set directories similar to 5.2
base_dir = '/home/chau/590-CN490/HW4.0/DOGS-AND-CATS/Data_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test1')

# Rescale images
datagen = ImageDataGenerator(rescale=1./255)
# Set batch_size
batch_size = BATCH_SIZE

# ---------------------------------------------------------------------------- #
# Define function to extract features
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels
# ---------------------------------------------------------------------------- #
# Define function to draw smooth curve
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
              previous = smoothed_points[-1]
              smoothed_points.append(previous * factor + point * (1 - factor))
        else:
              smoothed_points.append(point)
    return smoothed_points


# ---------------------------------------------------------------------------- #

#Print
print("Train data: ")
train_features, train_labels = extract_features(train_dir, 2000)

print("Validation data: ")
validation_features, validation_labels = extract_features(validation_dir, 1000)

print("Test data: ")
test_features, test_labels = extract_features(test_dir, 1000)

# ---------------------------------------------------------------------------- #
# Flatten features before connecting to densely-connected layer
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))


# ---------------------------------------------------------------------------- #

if (RUN_FEATURE_EXTRACT_PRETRAINED == 1):

    # Initiate model
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    # Use drop out for regularization
    model.add(layers.Dropout(0.5))
    # Sigmoid activatin for output layer because binary classification
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                loss='binary_crossentropy',
                metrics=['acc'])

    # Save the model
    history = model.fit(train_features, train_labels,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(validation_features, validation_labels))

    model.save('./Models/cats_and_dogs_small_5.3_feature_extract_conv_base.h5')
    # ---------------------------------------------------------------------------- #
    # Plot it

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show(block = False)

# ---------------------------------------------------------------------------- #
#                   FEATURE EXTRACTION WITH DATA AUGMENTATION                  #
# ---------------------------------------------------------------------------- #

if (RESOURCE == "CPU"):
    print("No GPU")
    print("Not running feature extraction with data augmentation")

if (RESOURCE == "CPU_GPU") & (RUN_FEATURE_EXTRACT_DATA_AUG == 0):
    print("Has GPU")
    print("Not running feature extraction with data augmentation")


if (RESOURCE == "CPU_GPU") & (RUN_FEATURE_EXTRACT_DATA_AUG ==1):
    print("Has GPU")
    print("Trying feature extraction using data augmentation with GPU")
    
    model = models.Sequential()
    model.add(conv_base) #Add conv_base like a layer
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))

    # Freeze the network
    conv_base.trainable = False

    print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

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
    
    validaton_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=BATCH_SIZE,
        class_mode='binary')
    
    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['acc'])

    history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=VAL_STEPS,
        verbose=1)
    
    model.save('./Models/cats_and_dogs_small_5.3_feature_extract_data_aug_GPU.h5')
    
# ---------------------------------------------------------------------------- #
# Plot it
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show(block=False)
    # ---------------------------------------------------------------------------- #
    
# ---------------------------------------------------------------------------- #
#                                  FINE TUNING                                 #
# ---------------------------------------------------------------------------- #

if (RUN_FINE_TUNING == 1):
    
    print("Fine tuning")
    
    conv_base.trainable = True

    # Freeze all layers up to block4_pool
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    
    #Bypassing the GPU feature extraction with data augmentation so need to remake the generators
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            # This is the target directory
            train_dir,
            # All images will be resized to 150x150
            target_size=(150, 150),
            batch_size=20,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')
    
    # ---------------------------------------------------------------------------- #
    # Load the model from feature extraction with data augmentation on GPU
    model = load_model('./Models/cats_and_dogs_small_5.3_feature_extract_data_aug_GPU.h5',compile=True)
 
    
    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-5),
                metrics=['acc'])

    history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=VAL_STEPS)

    #Save
    model.save('./Models/cats_and_dogs_small_5.3_fine_tuning.h5')
# ---------------------------------------------------------------------------- #

    #Plot it
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    
    plt.plot(epochs,
            smooth_curve(acc), 'bo', label='Smoothed training acc')
    plt.plot(epochs,
            smooth_curve(val_acc), 'b', label='Smoothed validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs,
            smooth_curve(loss), 'bo', label='Smoothed training loss')
    plt.plot(epochs,
            smooth_curve(val_loss), 'b', label='Smoothed validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

# ---------------------------------------------------------------------------- #
# Use model on test data

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=BATCH_SIZE,
        class_mode='binary')

test_loss, test_acc = model.evaluate(test_generator, steps=50)
print('test acc:', test_acc)