import os
# ---------------------------------------------------------------------------- #
# TEST = 'UNIVERSE'
TEST = 'AUTHOR'
# ---------------------------------------------------------------------------- #
TF_LOG_LEVEL = '2'
# ---------------------------------------------------------------------------- #
MODEL = 'CONV1D'
# MODEL = 'SRNN'
# MODEL = 'LSTM'
# ---------------------------------------------------------------------------- #
# RERUN_CLEAN = 1
RERUN_CLEAN = 0

# RERUN_TRAIN = 1
RERUN_TRAIN = 0

# ---------------------------------------------------------------------------- #
chunk_size = 4 # How many sentences in a text chunk
maxlen = 1000 # cut chunk off after how many words
max_words = 10000 # consider top 10k words in the dataset
training_split = 0.8
# ---------------------------------------------------------------------------- #
EPOCHS = 50
BATCH_SIZE = 128
# ---------------------------------------------------------------------------- #
SCORE_TO_MONITOR = 'val_acc'; MODE = 'max'
#SCORE_TO_MONITOR = 'val_loss'; MODE = 'min'


# ---------------------------------------------------------------------------- #
processed_path = './Processed_data/'

model_path = "./Models"
model_save_path = os.path.join(model_path, MODEL)

best_model_path = model_save_path+'_'+ MODE+'_'+SCORE_TO_MONITOR+'.hdf5'
best_model_training_scores_path = model_save_path+'_'+ MODE+'_'+SCORE_TO_MONITOR+'_training_scores.npy'
# ---------------------------------------------------------------------------- #


# print(history_save_path)
# exit()
plot_path = "./Plots"
plot_save_path = os.path.join(plot_path, MODEL)

