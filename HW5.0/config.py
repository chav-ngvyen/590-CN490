import os
# ---------------------------------------------------------------------------- #
# TEST = 'CORPUS'
# TEST = 'UNIVERSE'
# TEST = 'AUTHOR'
# ---------------------------------------------------------------------------- #
TF_LOG_LEVEL = '2'
# ---------------------------------------------------------------------------- #
MODEL_LIST = ['CONV1D','LSTM','LSTM-BIDIRECTIONAL','GRU','CNN_TO_RNN']
# MODEL_LIST = ['CONV1D', 'LSTM-BIDIRECTIONAL','CNN_TO_RNN']
# ---------------------------------------------------------------------------- #
RERUN_CLEAN = 1
# RERUN_CLEAN = 0

RERUN_TRAIN = 1
# RERUN_TRAIN = 0

I_PLOT = 1
# ---------------------------------------------------------------------------- #
chunk_size = 4 # How many sentences in a text chunk
maxlen = 250 # maximum size of a sequence
max_features = 10000 # size of dictionary
embed_dim = 16

training_split = 0.8
val_split = 0.15
# ---------------------------------------------------------------------------- #
EPOCHS = 50
BATCH_SIZE = 32
L2 = 1e-6
# ---------------------------------------------------------------------------- #
SCORE_TO_MONITOR = 'val_acc'; MODE = 'max'
# SCORE_TO_MONITOR = 'val_loss'; MODE = 'min'


# ---------------------------------------------------------------------------- #
processed_path = './Processed_data/'


