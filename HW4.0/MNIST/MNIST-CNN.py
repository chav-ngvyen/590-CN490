import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.models import load_model

#-------------------------
#USER PARAM 
#-------------------------

# DATASET     =   'MNIST'      
DATASET     =   'FASHION MNIST'   
# DATASET     =   'CIFAR10'   

# MODEL_TYPE  = "DFF"
MODEL_TYPE  = "CNN"

# CONV LAYER PARAM
FILTERS     =   [32,64,64,128]
N_CONV      = len(FILTERS)
if (MODEL_TYPE == "DFF"): N_CONV = 0

#DENSE FEED FORWARD PARAM
N_DENSE     =   3               #NUMBER OF FULLY CONNECTED LAYERS
N_NODES  	=	FILTERS[-1]     #NUMBER OF NODES IS THE LAST NUMBER OF FILTERS FOR CONVOLUTIONAL PART

ACT_TYPE    =   'relu'          #ACTIVATION FUNTION FOR HIDDEN LAYERS


#TRAINING PARAM
FRAC_KEEP   =   0.5        #SCALE DOWN DATASET FOR DEBUGGGING 
FRAC_BATCH  =   0.01        #CONTROLS BATCH SIZE
OPTIMIZER	=	'rmsprop' 
LR          =   0           #ONLY HAS EFFECT WHEN OPTIMIZER='rmsprop' (IF 0 USE DEFAULT)
L2_CONSTANT =   0.0         #IF 0 --> NO REGULARIZATION (DEFAULT 1e-4)

#DATA AUGMENTATION
DATA_AUG    =   1           #IF 1 --> REGULARIZATION WITH AUGMENTATION

EPOCHS 		= 	10
N_KFOLD     =   1           #NUM K FOR KFOLD (MAKE 1 FOR SINGLE TRAINING RUN)
VERSBOSE    =   1
NORM        = "NONE"
VAL_SIZE    =   0.2         #SIZE OF VALIDATION SPLIT
#-------------------------
#GET DATA AND DEFINE PARAM
#-------------------------

if(DATASET=='MNIST'):
    METRICS                     = ['accuracy']
    LOSS                        = 'categorical_crossentropy'
    OUTPUT_ACTIVATION           = 'softmax'
    (x, y), (x_test, y_test)    = mnist.load_data()
    RESHAPE                     = 1
    MODEL_SAVE_PATH             = './Models/MNIST/'
    
if(DATASET=='FASHION MNIST'):
    METRICS                     = ['accuracy']
    LOSS                        = 'categorical_crossentropy'
    OUTPUT_ACTIVATION           = 'softmax'
    (x, y), (x_test, y_test)    = fashion_mnist.load_data()
    RESHAPE                     = 1
    MODEL_SAVE_PATH             = './Models/FASHION_MNIST/'
   
if(DATASET=='CIFAR10'):
    METRICS                     = ['accuracy']
    LOSS                        = 'categorical_crossentropy'
    OUTPUT_ACTIVATION           = 'softmax'
    (x, y), (x_test, y_test)   =  cifar10.load_data()
    RESHAPE                     = 0
    MODEL_SAVE_PATH             = './Models/CIFAR10/'




# ---------------------------------------------------------------------------- #
def explore(x,y,tag=''):
    print("------EXPLORE RAW "+tag.upper()+" DATA------")
    print("x type:",x.shape)
    print("x type:",type(x),type(x[0]))  
    print("y shape:",y.shape)
    print("y type:",type(y),type(y[0]))
    for i in range(0,5):
        if(str(type(x[i]))=="<class 'numpy.ndarray'>"):
            print(" x["+str(i)+"] shape:",x[i].shape, "y[i]=",y[i]) 
        if(str(type(x[i]))=="<class 'list'>"):
            print(" x["+str(i)+"] len:",len(x[i]), "y[i]=",y[i]) 

explore(x,y,"TRAIN")
explore(x_test,y_test,"TEST")
# exit()
#-------------------------
#DATA PREPROCESSING/NORM 
#-------------------------

#DOWNSIZE DATASET FOR DEBUGGING IF DESIRED
if(FRAC_KEEP<1.0):
    NKEEP=int(FRAC_KEEP*x.shape[0])
    rand_indices = np.random.permutation(x.shape[0])
    x=x[rand_indices[0:NKEEP]]
    y=y[rand_indices[0:NKEEP]]

#RESHAPE FOR GRAYSCALE DATASETS
#Add additional array of 1 for grayscale
if (RESHAPE == 1):
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

#RESHAPE INPUT DATA FOR DFF
#Because DFF needs a matrix of shape (samples, batch*height*width)
if (MODEL_TYPE == "DFF"):
    x = np.reshape(x, (x.shape[0],x.shape[1]*x.shape[2]*x.shape[3]))
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3]))

#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
tmp=y[0]
y = to_categorical(y)
y_test = to_categorical(y_test)
print(tmp, '-->',y[0])
print("train_labels shape:", y.shape)

#NORMALIZE TO 0-1:
x = x.astype('float32') / 255 
x_test = x_test.astype('float32') / 255  

#-----------------
#MODEL
#-----------------

#BUILD LAYER ARRAYS FOR ANN
ACTIVATIONS=[]; LAYERS=[]   
for i in range(0,N_CONV+N_DENSE):
    LAYERS.append(N_NODES)
    ACTIVATIONS.append(ACT_TYPE)

print("LAYERS:",LAYERS)
print("ACTIVATIONS:", ACTIVATIONS)

# # ---------------------------------------------------------------------------- #
#-----------------
#BUILD MODEL
#-----------------

def build_model():
    
    model = models.Sequential()
    if (MODEL_TYPE == "DFF"):
        
        model.add(layers.Dense(LAYERS[0], activation=ACTIVATIONS[0],
                               input_shape=(x.shape[1],),
                               kernel_regularizer=regularizers.l2(L2_CONSTANT)))

    if (MODEL_TYPE == "CNN"):
        
        model.add(layers.Conv2D(filters = FILTERS[0],
                                kernel_size = (3, 3),
                                activation = ACTIVATIONS[0],
                                input_shape = (x.shape[1], x.shape[2], x.shape[3])))
        
        model.add(layers.MaxPooling2D((2,2)))

        for i in range(1, len(FILTERS)):
            # model.add(layers.MaxPooling2D((2,2)))
            model.add(layers.Conv2D(filters = FILTERS[i],
                                    kernel_size = (3, 3),
                                    activation = ACTIVATIONS[i]))
        model.add(layers.Flatten())

    for i in range(1, N_DENSE):
        model.add(layers.Dense(units = FILTERS[-1], activation = ACTIVATIONS[-1]))
    # OUTPUT LAYER
    model.add(layers.Dense(y.shape[1], activation=OUTPUT_ACTIVATION))


    #COMPILE
    if(OPTIMIZER=='rmsprop' and LR!=0):
        opt = optimizers.RMSprop(learning_rate=LR)
    else:
        opt = OPTIMIZER

    model.compile(
    optimizer=opt, 
    loss=LOSS, 
    metrics=METRICS
                 )
    return model 
        
        
model = build_model()
model.summary()

#-----------------
#TRAIN MODEL
#-----------------

samples_per_k = x.shape[0] // N_KFOLD
if(N_KFOLD==1): samples_per_k=int(VAL_SIZE*x.shape[0])
# print(N_KFOLD,samples_per_k,x.shape,y.shape)

#ADD REGULARIZERS + LR

#RANDOMIZE ARRAYS
rand_indx = np.random.permutation(x.shape[0])
x=x[rand_indx]; y=y[rand_indx]

#LOOP OVER K FOR KFOLD
for k in range(0,N_KFOLD):

    print('---PROCESSING FOLD #', k,"----")
    
    #VALIDATION: (SLIDING WINDOW LEFT TO RIGHT)
    x_val = x[k * samples_per_k: (k + 1) * samples_per_k]
    y_val = y[k * samples_per_k: (k + 1) * samples_per_k]

    #TRAINING: TWO WINDOWS (LEFT) <--VAL--> (RIGHT)
    x_train = np.concatenate(
        [x[:k * samples_per_k],
         x[(k + 1) * samples_per_k:]],
        axis=0)
    
    y_train = np.concatenate(
        [y[:k * samples_per_k],
         y[(k + 1) * samples_per_k:]],
        axis=0)

    #PRINT TO SEE WHAT LOOP IS DOING model: T= {}'.format(t) model: T= {}'.format(t)
    print("A",k * samples_per_k,(k + 1) * samples_per_k)
    print("B",x[:k * samples_per_k].shape, x[(k + 1) * samples_per_k:].shape)
    print("C",x_train.shape,y_train.shape)
    print("D",x_val.shape,y_val.shape)
    BATCH_SIZE=int(FRAC_BATCH*x_train.shape[0])

    #BUILD MODEL 
    model = build_model()
    if(k==0):  model.summary()

    #FIT MODEL
    history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=VERSBOSE,
    validation_data=(x_val, y_val)
    )
    
    #SAVE MODEL
    model.save(os.path.join(MODEL_SAVE_PATH, "model.h5"))
    print("Model saved")
    #------------------
    # DATA AUGMENTATION
    #------------------
    #Partly adapted from this https://www.novatec-gmbh.de/en/blog/keras-data-augmentation-for-cnn/

    if (DATA_AUG == 1):
        print("DATA AUGMENTATION STEP")
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Add augmentation to the data
        gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.5,
            zoom_range=(0.9, 1.1),
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode='constant',
            cval=0)

        # Put data in generator
        train_generator = gen.flow(x_train, 
                                y_train, 
                                batch_size=BATCH_SIZE)
    
        # Rebuild the model
        model = build_model()
 
        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=VERSBOSE,
            validation_data=(x_val, y_val)
            )    

        model.save(os.path.join(MODEL_SAVE_PATH, "model_data_aug.h5"))
        print("Model with data aug saved")

    #BASIC PLOTTING 
    I_PLOT=False
    if(k==N_KFOLD-1): I_PLOT=True                
    if(I_PLOT):
        #LOAD THE CORRECT MODEL BASED ON DATA AUG
        if (DATA_AUG == 1):
            model = load_model(os.path.join(MODEL_SAVE_PATH, "model_data_aug.h5"), compile = True) 
        else:
            model = load_model(os.path.join(MODEL_SAVE_PATH, "model.h5"), compile = True) 
             
        #LOSS 
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'rx', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        #METRICS
        if(len(METRICS)>0):
            for metric in METRICS:
                plt.clf()
                MT = history.history[metric]
                MV = history.history['val_'+metric]
                plt.plot(epochs, MT, 'bo', label='Training '+metric)
                plt.plot(epochs, MV, 'rx',  label='Validation '+metric)
                plt.title('Training and validation '+metric)
                plt.xlabel('Epochs')
                plt.ylabel(metric)
                plt.legend()
                plt.show()

    train_values  = model.evaluate(x_train, y_train,batch_size=y_val.shape[0],verbose=VERSBOSE)
    val_values   = model.evaluate(x_val,   y_val,batch_size=y_val.shape[0],verbose=VERSBOSE)

    # scores.append(val_mae)
    print("--------------------------")
    print("RESULTS FOR K=",k)
    print("TRAIN:",train_values)
    print("VALIDATION:",val_values)
    print("--------------------------")

    if(k==0):
        train_ave=np.array(train_values)
        val_ave=np.array(val_values)
    else:
        train_ave=train_ave+np.array(train_values)
        val_ave=val_ave+np.array(val_values)

#AVERAGE
train_ave=train_ave/N_KFOLD
val_ave=val_ave/N_KFOLD
test_values = model.evaluate(x_test, y_test,batch_size=y_val.shape[0])

#FINAL REPORT
print("--------------------------")
print("AVE TRAIN:",train_ave)
print("AVE VALIDATION:",val_ave)
print("TEST:",test_values)
# print("VAL  RATIOS",np.array(train_ave)/np.array(val_ave))
# print("TEST RATIOS",np.array(train_values)/np.array(test_values))
print("--------------------------")




exit()
