# Doyle, Christie or LeBlanc? 

Train a neural network to determine whether a chunk of text was written by Arthur Conan Doyle, Agatha Christie and Maurice LeBlanc. I test the neural net on parts of Holmes/ Poirot/ Lupin stories not in the training set to determine if it can tell the three authors apart from a short chunk of text. After this works, I plan to test the model on texts written by the authors about different characters/ settings (The Lost World - Doyle, for example) to see if it can determine the ``penmanship'' of the authors when they write about other characters/ topics. 

## HOW TO RUN:
Step 1: Change hyperparameters in config.py; save it

Step 2: bash RUN.sh



### Codes:
#### RUN.sh
Run this to execute 01-clean, 02-train, and 03-config. Remember to change the parameters in RERUN_CLEAN and RERUN_TRAIN in config.py to rerun cleaning and training scripts.
Also saves the terminal output to ./Logs
#### config.py
This is the script to change script 01 cleaning options, does hyperparameter tuning in script 02, and evaluate different models in script 03.  
#### 01-clean.py
Cleans the raw novels in downloaded from Project Gutenberg in .txt format, split each novel into chunks of n sentences, tokenize the text sequences using keras' built-in text processing tool and save the training and test data as .npy files
#### 02-train.py
Train the neural network using inputs from config.py, save models/
#### 03-evaluate.py
Evaluate the models, make and save plots

### Folders:
#### Raw_books
Raw txt files from Project Gutenberg
#### Prcossed_data
Training, validation and test data saved in npy format after 01-clean.py is done running
#### Models, Plots, Logs
Stores outputs from 02-train and 03-evaluate