import config

def main():
    RERUN = config.RERUN_TRAIN
    if RERUN == 0: 
        print("Not rerunning training script")
        return 
    
    else:
        import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_LOG_LEVEL
        print("TF log level =", os.environ['TF_CPP_MIN_LOG_LEVEL'])
        import tensorflow as tf 

        import numpy as np
        np.random.seed(42)

        import regex as re

        import os.path

        import matplotlib.pyplot as plt

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import SimpleRNN, LSTM, Embedding
        from tensorflow.keras import layers
        from tensorflow.keras.optimizers import RMSprop
        from tensorflow.keras.callbacks import ModelCheckpoint

        from config import MODEL_LIST

        # Print config details
        print("\nConfig:")
        maxlen = config.maxlen; print("Max length: ", maxlen)
        chunk_size = config.chunk_size; print("Sentences per chunk: ", chunk_size)
        max_features = config.max_features; print("Max features: ", max_features)
        embed_dim = config.embed_dim; print("Embedded dimension", embed_dim)
        
        training_split = config.training_split; print("Training split: ", training_split)

        print("\nHyper-parameters")
        MODEL_LIST = config.MODEL_LIST
        EPOCHS=config.EPOCHS; print("Epochs: ", EPOCHS)
        BATCH_SIZE=config.BATCH_SIZE; print("Batch size: ", BATCH_SIZE)
        L2 = config.L2 

        # ---------------------------------------------------------------------------- #
        # Load data cleaned from 01-clean.py

        print("\nLoading clean data:")
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
        # Plotting functions

        def report(history,title='',I_PLOT=True):
    
            print(title+": CORPUS TEST METRIC (loss,accuracy):",model.evaluate(x_test_CORPUS,y_test_CORPUS,batch_size=BATCH_SIZE,verbose=1))
            print(title+": UNIVERSE TEST METRIC (loss,accuracy):",model.evaluate(x_test_UNIVERSE,y_test_UNIVERSE,batch_size=BATCH_SIZE,verbose=1))
            print(title+": AUTHOR TEST METRIC (loss,accuracy):",model.evaluate(x_test_AUTHOR,y_test_AUTHOR,batch_size=BATCH_SIZE,verbose=1))

            if(I_PLOT):
                #PLOT HISTORY
                epochs = range(1, len(history.history['loss']) + 1)
                plt.figure()
                plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
                plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
                plt.title(title)
                plt.legend()
                plt.savefig('./Plots/'+title+'_LOSS_'+config.MODE+'_'+config.SCORE_TO_MONITOR+'.png')
                plt.clf()
                
                plt.plot(epochs, history.history['acc'], 'ro', label='Training acc')
                plt.plot(epochs, history.history['val_acc'], 'r', label='Validation acc')
                plt.title(title)
                plt.legend()
                plt.savefig('./Plots/'+title+'_ACC_'+config.MODE+'_'+config.SCORE_TO_MONITOR+'.png')
                plt.close()
        
        #-------------------------------------------------------------------- #
        # Define models

        for m in MODEL_LIST:
            MODEL = m

            if (MODEL == 'CONV1D'):
                print("--------------")
                print(MODEL)
                print("--------------")
                model = Sequential()
                model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
                model.add(layers.Conv1D(32, 7, activation='relu'))
                model.add(layers.MaxPooling1D(5))
                model.add(layers.Conv1D(32, 7, activation='relu'))
                model.add(layers.GlobalMaxPooling1D())
                model.add(layers.Dense(3, activation = 'softmax'))
                model.summary()

            if (MODEL == 'LSTM'):
                print("---------------------------")
                print(MODEL)  
                print("---------------------------")

                model = Sequential() 
                model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
                model.add(layers.LSTM(BATCH_SIZE, recurrent_regularizer=tf.keras.regularizers.l2(L2))) 
                model.add(layers.Dense(3, activation='softmax'))
                model.summary()
        
            if (MODEL == 'LSTM-BIDIRECTIONAL'):
                print("---------------------------")
                print("LSTM-BIDIRECTIONAL")  
                print("---------------------------")

                model = Sequential() 
                model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
                model.add(layers.Bidirectional(layers.LSTM(BATCH_SIZE, recurrent_regularizer=tf.keras.regularizers.l2(L2)))) 
                model.add(layers.Dense(3, activation='softmax'))
                model.summary()

            if (MODEL == 'GRU'):
                print("---------------------------")
                print(MODEL)  
                print("---------------------------")

                model = Sequential() 
                model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
                model.add(layers.GRU(BATCH_SIZE)) 
                model.add(layers.Dense(3, activation='softmax'))
                model.summary()
            
            if (MODEL == 'CNN_TO_RNN'):
                print("---------------------------")
                print(MODEL)  
                print("---------------------------")

                model = Sequential()
                model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
                model.add(layers.Conv1D(32, 7, activation='relu'))
                model.add(layers.MaxPooling1D(5))
                model.add(layers.Conv1D(32, 7, activation='relu'))
                model.add(layers.GRU(32 , dropout=0.1, recurrent_dropout=0.5))
                model.add(layers.Dense(3, activation = 'softmax'))
                model.summary() 
                
            # ---------------------------------------------------------------------------- #
            # Compile the model
            model.compile(optimizer=RMSprop(learning_rate=1e-4),
                        loss='sparse_categorical_crossentropy',
                        metrics=['acc'])

            # Create callbacks
            checkpoint = ModelCheckpoint(filepath="./Models/"+MODEL+'_'+ config.MODE+'_'+config.SCORE_TO_MONITOR+'.hdf5', 
                                        monitor=config.SCORE_TO_MONITOR,
                                        verbose=1, 
                                        save_best_only=True,
                                        mode=config.MODE)
            CALLBACKS = [checkpoint]
                    
            # Fit model
            history = model.fit(x_train, y_train,
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE,
                                verbose = 1,
                                validation_data = (x_val, y_val),
                                callbacks = CALLBACKS)
            report(history, title=MODEL)

if __name__ == "__main__":
    main()
 