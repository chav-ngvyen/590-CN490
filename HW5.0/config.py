import os
# ---------------------------------------------------------------------------- #
# TEST = 'AUTHOR'; test_dir = './Test_books'
TEST = 'HOLMES'; test_dir = './Test_books2'
''' 
AUTHOR = Can model identify if author is LeBlanc, Christie, Doyle
HOLMES = Can model identify non-Holmes writings by Doyle or not?
'''
# ---------------------------------------------------------------------------- #
MODEL = 'CONV1D'
# MODEL = 'SRNN'
# MODEL = 'LSTM'
# ---------------------------------------------------------------------------- #
chunk_size = 4 # How many sentences in a text chunk
maxlen = 1000 # cut chunk off after how many words
max_words = 10000 # consider top 10k words in the dataset
training_split = 0.8
# ---------------------------------------------------------------------------- #
EPOCHS = 10
BATCH_SIZE = 128
# ---------------------------------------------------------------------------- #
model_path = "./Models"
model_save_path = os.path.join(model_path, MODEL)

plot_path = "./Plots"
plot_save_path = os.path.join(plot_path, MODEL)

