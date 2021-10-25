import config
import tensorflow as tf 
import numpy as np
np.random.seed(42)

import regex as re
from nltk import tokenize 
import os
import os.path

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


author_list = ['leblanc','doyle','christie']
train_dir = './Train_books'
test_dir = config.test_dir
# test_dir = './Test_books2'
data_dir = './Data'
# --------
chunk_size = config.chunk_size # How many sentences in a text chunk
maxlen = config.maxlen # cut chunk off after how many words
max_words = config.max_words # consider top 10k words in the dataset
training_split = config.training_split

EPOCHS=30
BATCH_SIZE = 128
# MODEL = "SRNN"
# MODEL = "LSTM"
MODEL = "CONV1D"
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
    dir_name = os.path.join(train_dir, author)
    if os.path.exists(dir_name) is not True:
        print("Path does not exist")
        pass
    else:
        for book_no, fname in enumerate(os.listdir(dir_name)):
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
        print("Book number: ", book_no, "Written by: ", author, "and has: ", len(chunks), "chunks.")
            
        for i in range(len(chunks)):
            texts.append(chunks[i])
            labels.append(author_no)


# # ---------------------------------------------------------------------------- #
        
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
training_samples = int(data.shape[0]*training_split)

x_train = data[:training_samples]
y_train = labels[:training_samples]

x_val = data[training_samples: ]
y_val = labels[training_samples: ]

np.save('./Data/x_train.npy', x_train)
np.save('./Data/y_train.npy', y_train)
np.save('./Data/x_val.npy', x_val)
np.save('./Data/y_val.npy', y_val)

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

np.save('./Data/x_test.npy', x_test)
np.save('./Data/y_test.npy', y_test)


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
