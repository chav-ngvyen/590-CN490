# Holmes, Poirot or Lupin? 
## A neural nets approach to Natural Language Processing and Authorship Detection

I train a neural network to determine whether a chunk of text was written by Arthur Conan Doyle, Agatha Christie and Maurice LeBlanc. I test the neural net on parts of Holmes/ Poirot/ Lupin stories not in the training set to determine if it can tell the three authors apart from a short chunk of text. Afterwards, I test the model on texts written by the authors about different characters/ settings (The Lost World - Doyle, for example) to see if it can determine the ``penmanship'' of the authors when they write about other characters/ topics. 

### Codes:
#### config.py
This is the script to change script 01 cleaning options, does hyperparameter tuning in script 02, and evaluate different models in script 03.  
#### 01-clean.py
Cleans the raw novels in downloaded from Project Gutenberg in .txt format, split each novel into chunks of n sentences, tokenize the text sequences using keras' built-in text processing tool and save the training and test data as .npy files
#### 02-train.py
Train the neural network using inputs from config.py, save models, plots and log files.
#### 03-evaluate.py
Evaluate the model

### Folders:
#### Train_books, Test_books, Test_books2
Raw txt files from Project Gutenberg
#### Data
Training and test data in npy format
#### Models, Plots, Logs
Stores outputs from 02-train and 03-evaluate