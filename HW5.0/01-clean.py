import config
# import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_LOG_LEVEL


        
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_LOG_LEVEL
import tensorflow as tf 
import os.path

import numpy as np
np.random.seed(42)

import regex as re
from nltk import tokenize 
# import os
# import os.path

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Embedding
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing import sequence
# ---------------------------------------------------------------------------- #
if config.RERUN_CLEAN == 0:
    print("\nNot re-running cleaning script")
    exit()
# elif config.RERUN_CLEAN == 0:
#     print("\nNot re-running cleaning script")
#     exit()

author_list = ['leblanc','doyle','christie']

train_books_path = './Raw_books/Train'
test_books_path = './Raw_books/Test'
processed_path = './Processed_data/'
# for i in ['_AUTHOR','_UNIVERSE']:
#     dir = train_books_path+i
#     print(dir)

# exit()
# train_dir = './Train_books'
# test_dir = ['./Test_books_AUTHOR','./Test_books_UNIVERSE']
# data_dir = './Data'
# --------
chunk_size = config.chunk_size # How many sentences in a text chunk
maxlen = config.maxlen # cut chunk off after how many words
max_words = config.max_words # consider top 10k words in the dataset
training_split = config.training_split

EPOCHS=config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
# MODEL = "SRNN"
# MODEL = "LSTM"
MODEL = config.MODEL
# ---------------------------------------------------------------------------- #
# Define function to divide text into chunks
# Where each chunk contains N sentences
# Note: Code adapted from Artificial Intelligence with Python (Joshi, Prateek)
def chunker(input_data, N):
    input_sentences = tokenize.sent_tokenize(input_data)
    output = []
    
    cur_chunk = []
    count = 0
    for sentence in input_sentences:
        cur_chunk.append(sentence)
        count += 1
        if count == N:
            output.append(' '.join(cur_chunk))
            count, cur_chunk = 0, []
            
    output.append(' '. join(cur_chunk))
    return output


# ---------------------------------------------------------------------------- #
labels = []
books = []
texts = []

for author_no, author in enumerate(author_list):
    dir_name = os.path.join(train_books_path, author)
    if os.path.exists(dir_name) is not True:
        print(dir_name, "Path does not exist")
        pass
    else:
        print("\n Processing train books in :", dir_name)
        for index, fname in enumerate(os.listdir(dir_name)):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))        
                raw = f.read()
                f.close()  
            # Get the real content of the book
            # Based on https://people.duke.edu/~ccc14/sta-663/TextProcessingExtras.html
            start = re.search(r"\*\*\* START OF ", raw).end()
            stop = re.search(r"\*\*\* END OF ", raw).start()    
            content = raw[start:stop]
            
            chunks = chunker(content, chunk_size)  
            print("Book ", index, "written by: ", author, "and has: ", len(chunks), "chunks.")
            
            for i in range(len(chunks)):
                texts.append(chunks[i])
                labels.append(author_no)


# # ---------------------------------------------------------------------------- #

print("\nDone processing, starting tokenizer") 
        
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' %len(word_index))
data = pad_sequences(sequences, maxlen = maxlen)

labels = np.asarray(labels)

# print(labels); exit()    
print('shape of data tensor:' , data.shape)
print('shape of label tensor: ', labels.shape)

# ---------------------------------------------------------------------------- #
# Shuffle the text chunks
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]
training_samples = int(data.shape[0]*training_split); print("\n Train/ Val split: ", training_split)

x_train = data[:training_samples]; print("\n Train x shape: ", x_train.shape)
y_train = labels[:training_samples]

x_val = data[training_samples: ]; print("\n Val x shape: ", x_val.shape)
y_val = labels[training_samples: ]

#np.save(processed_path+'/Train/x_train.npy', x_train)
# print(processed_path+'x_train.npy')
# exit()
np.save(processed_path+'x_train.npy', x_train)
np.save(processed_path+'y_train.npy', y_train)
np.save(processed_path+'x_val.npy', x_val)
np.save(processed_path+'y_val.npy', y_val)
# exit()
# ---------------------------------------------------------------------------- #
for test_type in ['_UNIVERSE', '_AUTHOR']:
    raw_test_dir = test_books_path+test_type
    
    # print(raw_test_dir)
    # exit()

    # Process the test data
    labels = []
    texts = []

    for author_no, author in enumerate(author_list):
        dir_name = os.path.join(raw_test_dir, author)
        if os.path.exists(dir_name) is not True:
            print(dir_name)
            print("Path does not exist")
            pass
        else:
            print("\nProcessing test books in :", dir_name)
            for index, fname in enumerate(os.listdir(dir_name)):
                if fname[-4:] == '.txt':
                    f = open(os.path.join(dir_name, fname))        
                    raw = f.read()
                    f.close()

                start = re.search(r"\*\*\* START OF .* \*\*\*", raw).end()
                stop = re.search(r"\*\*\* END OF ", raw).start()    
                content = raw[start:stop]
                
                chunks = chunker(content, chunk_size)  
                print("Book ", index, "written by: ", author, "and has: ", len(chunks), "chunks.")
                
                for i in range(len(chunks)):
                    texts.append(chunks[i])
                    labels.append(author_no)
        # exit()
       
    print("\nDone processing, starting tokenizer") 
    sequences = tokenizer.texts_to_sequences(texts)
    x_test = pad_sequences(sequences, maxlen = maxlen); print("Test x: ", x_test.shape)
    y_test = np.asarray(labels); print("Tesy y: ", y_test.shape)
    
    np.save(processed_path+'x_test'+test_type+'.npy', x_test)
    np.save(processed_path+'y_test'+test_type+'.npy', y_test)

    
    #print(labels); exit()
#     if (test_raw_dir == './Test_books_AUTHOR'):
#         test_save_dir = './Data/Test_AUTHOR/'
#     if (test_raw_dir == './Test_books_UNIVERSE'):
#         test_save_dir = './Data/Test_UNIVERSE/'

# np.save(test_save_dir+'x_test.npy', x_test)
# np.save(test_save_dir+'y_test.npy', y_test)


exit()

# ---------------------------------------------------------------------------- #
# Fit model

if (MODEL == "CONV1D" ):
    model = Sequential()
    model.add(layers.Embedding(max_words, BATCH_SIZE, input_length=maxlen))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(3, activation = 'softmax'))
    model.summary()

if (MODEL == "SRNN"):
    model = Sequential()
    model.add(layers.Embedding(max_words, BATCH_SIZE))
    model.add(layers.SimpleRNN(BATCH_SIZE))
    model.add(layers.Dense(3, activation='softmax'))
    model.summary()

if (MODEL == "LSTM"):
    model = Sequential()
    model.add(layers.Embedding(max_words, BATCH_SIZE))
    model.add(layers.LSTM(BATCH_SIZE))
    model.add(layers.Dense(3, activation = 'softmax'))
    model.summary()

model.compile(optimizer=RMSprop(lr=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs=EPOCHS,
                    batch_size=128,
                    verbose = 1,
                    validation_data = (x_val, y_val))
# ---------------------------------------------------------------------------- #


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# ---------------------------------------------------------------------------- #
# Process the test data
print("Processing test data")
labels = []
texts = []

for author_no, author in enumerate(author_list):
    dir_name = os.path.join(test_dir, author)
    if os.path.exists(dir_name) is not True:
        print("Path does not exist")
        pass
    else:
        for book_no, fname in enumerate(os.listdir(dir_name)):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))        
                raw = f.read()
                f.close()

        start = re.search(r"\*\*\* START OF .* \*\*\*", raw).end()
        stop = re.search(r"\*\*\* END OF ", raw).start()    
        content = raw[start:stop]
            
        chunks = chunker(content, chunk_size)  
        print("Book number: ", book_no, "Written by: ", author, "and has: ", len(chunks), "chunks.")
            
        for i in range(len(chunks)):
            texts.append(chunks[i])
            labels.append(author_no)
            
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen = maxlen)
y_test = np.asarray(labels)
# ---------------------------------------------------------------------------- #
model.evaluate(x_test, y_test)
exit()
