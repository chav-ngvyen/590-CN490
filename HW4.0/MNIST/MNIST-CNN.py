

# import warnings
# warnings.filterwarnings("ignore")

import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import matplotlib.pyplot as plt


#-------------------------
#USER PARAM 
#-------------------------

DATASET     =   'MNIST'      
# DATASET     =   'FASHION MNIST'   
# DATASET     =   'CIFAR10'   

# MODEL_TYPE  = "DFF"
MODEL_TYPE  = "CNN"

# CONV LAYER PARAM
FILTERS     =   [32,64,64,128]
N_CONV      =   len(FILTERS)

#DENSE FEED FORWARD PART PARAM
N_DENSE    =   2           #NUMBER OF FULLY CONNECTED LAYERS
N_NODES  	=	FILTERS[-1]          #NUMBER OF NODES IS THE LAST NUMBER OF FILTERS FOR CONVOLUTIONAL PART

ACT_TYPE    =   'relu'      #ACTIVATION FUNTION FOR HIDDEN LAYERS


#TRAINING PARAM
FRAC_KEEP   =   0.01         #SCALE DOWN DATASET FOR DEBUGGGING 
FRAC_BATCH  =   0.01        #CONTROLS BATCH SIZE
OPTIMIZER	=	'rmsprop' 
LR          =   0           #ONLY HAS EFFECT WHEN OPTIMIZER='rmsprop' (IF 0 USE DEFAULT)
L2_CONSTANT =   0.0         #IF 0 --> NO REGULARIZATION (DEFAULT 1e-4)

EPOCHS 		= 	1
N_KFOLD     =   1           #NUM K FOR KFOLD (MAKE 1 FOR SINGLE TRAINING RUN)
VERSBOSE    =   1
NORM        = "NONE"

#-------------------------
#GET DATA AND DEFINE PARAM
#-------------------------

if(DATASET=='MNIST'):
    METRICS                     = ['accuracy']
    LOSS                        = 'categorical_crossentropy'
    OUTPUT_ACTIVATION           = 'softmax'
    (x, y), (x_test, y_test)    = mnist.load_data()
    RESHAPE                     = 1
    
if(DATASET=='FASHION MNIST'):
    METRICS                     = ['accuracy']
    LOSS                        = 'categorical_crossentropy'
    OUTPUT_ACTIVATION           = 'softmax'
    (x, y), (x_test, y_test)    = fashion_mnist.load_data()
    RESHAPE                     = 1
    
if(DATASET=='CIFAR10'):
    METRICS                     = ['accuracy']
    LOSS                        = 'categorical_crossentropy'
    OUTPUT_ACTIVATION           = 'softmax'
    (x, y), (x_test, y_test)   =  cifar10.load_data()
    RESHAPE                     = 0
    
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

# if(DATASET=="IMDB" or DATASET=="REUTERS"):
#     # Encoding [3, 5] --> [0,0,0,1,0,1,0 .... ]
#     def vectorize_sequences(sequences, dimension):
#         results = np.zeros((len(sequences), dimension))
#         for i, sequence in enumerate(sequences):
#             results[i, sequence] = 1.
#         return results

#     x = vectorize_sequences(x,NUM_WORDS)
#     x_test = vectorize_sequences(x_test,NUM_WORDS)

# if(DATASET=="IMDB"):   #BINARY DATA
#     y = np.asarray(y).astype("float32")
#     y_test = np.asarray(y_test).astype("float32")

# if(DATASET=="REUTERS"): #MULTI-LABEL DATA
#     # 3 --> [0,0,0,1,0,0, ...]
#     from keras.utils.np_utils import to_categorical
#     y = to_categorical(y)
#     y_test = to_categorical(y_test)


#DOWNSIZE DATASET FOR DEBUGGING IF DESIRED
if(FRAC_KEEP<1.0):
    NKEEP=int(FRAC_KEEP*x.shape[0])
    rand_indices = np.random.permutation(x.shape[0])
    x=x[rand_indices[0:NKEEP]]
    y=y[rand_indices[0:NKEEP]]

# ---------------------------------------------------------------------------- #
#RESHAPE FOR GRAYSCALE DATASETS
if (RESHAPE == 1):
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

#RESHAPE INPUT DATA FOR DFF
if (MODEL_TYPE == "DFF"):
    x = np.reshape(x, (x.shape[0],x.shape[1]*x.shape[2]*x.shape[3]))
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3]))

#NORMALIZE TO 0-1:
x = x.astype('float32') / 255 
x_test = x_test.astype('float32') / 255  


#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
tmp=y[0]
y = to_categorical(y)
y_test = to_categorical(y_test)
print(tmp, '-->',y[0])
print("train_labels shape:", y.shape)

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
        #model.add(layers.Dense(units = FILTERS[-1], activation = ACTIVATIONS[-1]))
    # layer_names = []
    # for layer in model.layers[:]:
    #     layer_names.append(layer.name)
        
    # for layer_name, layer_activation in zip(layer_names, ACTIVATIONS):
    #     n_features = layer_activation.shape[-1]
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


# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# ---------------------------------------------------------------------------- #
#BUILD KERAS MODEL
# def build_model():
#     model = models.Sequential()
    
#     model.add(layers.Flatten())        
#     #HIDDEN LAYERS
#     model.add(layers.Dense(LAYERS[0], activation=ACTIVATIONS[0], input_shape=(x_train.shape[1],), kernel_regularizer=regularizers.l2(L2_CONSTANT)))
#     for i in range(1,len(LAYERS)):
#         model.add(layers.Dense(LAYERS[i], activation=ACTIVATIONS[i], kernel_regularizer=regularizers.l2(L2_CONSTANT)))
#     #OUTPUT LAYER
#     model.add(layers.Dense(y.shape[1], activation=OUTPUT_ACTIVATION, kernel_regularizer=regularizers.l2(L2_CONSTANT)))

#     #COMPILE
#     if(OPTIMIZER=='rmsprop' and LR!=0):
#         opt = optimizers.RMSprop(learning_rate=LR)
#     else:
#         opt = OPTIMIZER

#     model.compile(
#     optimizer=opt, 
#     loss=LOSS, 
#     metrics=METRICS
#                  )
#     return model 


#-----------------
#TRAIN MODEL
#-----------------

samples_per_k = x.shape[0] // N_KFOLD
if(N_KFOLD==1): samples_per_k=int(0.2*x.shape[0])
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

    #PRINT TO SEE WHAT LOOP IS DOING
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

    #BASIC PLOTTING 
    I_PLOT=False
    if(k==N_KFOLD-1): I_PLOT=True                
    if(I_PLOT):

        #LOSS 
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
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
                plt.plot(epochs, MV, 'b',  label='Validation '+metric)
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

#PARITY PLOT
if(DATASET=="BOSTON"):
    fig, ax = plt.subplots()
    ax.plot(y , model.predict(x),'o', label='Train+Val') 
    ax.plot(y_test  , model.predict(x_test),'*', label='Test') 
    plt.xlabel('y_pred', fontsize=18);   plt.ylabel('y_data', fontsize=18);   plt.legend()
    plt.show()


exit()

# x_train = vectorize_sequences(train_data)
# # print(type(x_train),x_train.shape)
# # print(x_train[0])
# # print(x_train[1])
# # exit()

# x_test = vectorize_sequences(test_data)


# def to_one_hot(labels, dimension=46):
#     results = np.zeros((len(labels), dimension))
#     for i, label in enumerate(labels):
#         results[i, label] = 1.
#     return results

# one_hot_train_labels = to_one_hot(train_labels)
# print(one_hot_train_labels.shape); 
# print(train_labels[0],one_hot_train_labels[0]); 
# print(train_labels[1],one_hot_train_labels[1]); 

# one_hot_test_labels = to_one_hot(test_labels)

# OUT_DIM=one_hot_train_labels.shape[1]
# print(IN_DIM,OUT_DIM)



# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)










# x_val = x_train[:1000]
# partial_x_train = x_train[1000:]
# y_val = one_hot_train_labels[:1000]
# partial_y_train = one_hot_train_labels[1000:]







#MANUALLY ENTER 
# LAYERS        =   [24,24,24]
# ACTIVATIONS   =   ['relu','relu','relu']


# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(46, activation='softmax'))
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(partial_x_train,
#           partial_y_train,
#           epochs=9,
#           batch_size=512,
#           validation_data=(x_val, y_val))
# results = model.evaluate(x_test, one_hot_test_labels)



# import numpy as np
# v=np.random.uniform(low=0.0, high=1.0, size=5)
# print(v)
# p=np.exp(v)/sum(np.exp(v))
# print(p)
# print(sum(p))
# exit()





# from keras.datasets import boston_housing
# from keras import models
# from keras import layers
# import numpy as np


# # Load the data specify the predictor and reponse for training and testing
# # Data is read in as numpy arrays
# (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# # Get shape of each array
# print(train_data.shape)
# print(train_targets.shape)
# print(test_data.shape)
# print(test_targets.shape)

# mean = train_data.mean(axis=0)
# train_data -= mean
# std = train_data.std(axis=0)
# train_data /= std
# test_data -= mean
# test_data /= std


# def build_model():
#     model = models.Sequential()
#     model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
#     model.add(layers.Dense(64, activation='relu'))
#     model.add(layers.Dense(1))
#     model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
#     return model


# k=4
# num_val_samples = len(train_data) // k
# num_epochs = 100
# all_scores = []
# all_mae_histories = []

# for i in range(k):
#     print('processing fold #', i)
    
#     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    
#     val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
#     partial_train_data = np.concatenate(
#         [train_data[:i * num_val_samples],
#          train_data[(i + 1) * num_val_samples:]],
#         axis=0)
    
#     partial_train_targets = np.concatenate(
#         [train_targets[:i * num_val_samples],
#          train_targets[(i + 1) * num_val_samples:]],
#         axis=0)
    
#     model = build_model()
    
#     model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1)
#     val_mse, val_mae = model.evaluate(val_data, val_targets)
#     all_scores.append(val_mae)
    
#     # history = model.fit(partial_train_data, 
#     #                     partial_train_targets,
#     #                     validation_data=(val_data, val_targets),
#     #                     epochs=num_epochs, 
#     #                     batch_size=1,
#     #                     verbose=0)
    
#     # mae_history = history.history['val_mean_absolute_error']
    
#     # all_mae_histories.append(mae_history)
    
    
    
