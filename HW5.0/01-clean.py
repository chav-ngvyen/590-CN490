import config

def main():
    RERUN = config.RERUN_CLEAN
    if RERUN == 0:
        print("Not rerunning cleaning script")
        return 
    else:
        import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_LOG_LEVEL
        import tensorflow as tf 
        import os.path

        import numpy as np
        np.random.seed(42)

        import regex as re
        import string
        from nltk import tokenize 

        import matplotlib.pyplot as plt

        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        author_list = ['leblanc','doyle','christie']

        train_books_path = './Raw_books/Train'
        test_books_path = './Raw_books/Test'
        processed_path = './Processed_data/'

        # --------
        chunk_size = config.chunk_size # How many sentences in a text chunk
        maxlen = config.maxlen # maximum size of a sequence
        max_features = config.max_features # size of dictionary to use
        
        training_split = config.training_split
        val_split = config.val_split
        

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
                        
                    # Get the real content of the book between START AND END
                    # Based on https://people.duke.edu/~ccc14/sta-663/TextProcessingExtras.html
                    try:
                        start = re.search(r"\*\*\*.*START OF .*GUTENBERG.*\*\*\*", raw).end()
                        stop = re.search(r"\*\*\*.*END OF .*GUTENBERG.*\*\*\*", raw).start()    
                        content = raw[start:stop]
                    
                    except AttributeError:
                        print("Check raw file for", fname)
                        content = raw

                    # content = content.splitlines()
                    # Printable characters only
                    printable = set(string.printable); tmp=''
                    for char in content:
                        if (char in printable): tmp=tmp+char
                    content = tmp
                    
                    
                    chunks = chunker(content, chunk_size)  
                    print(fname, "written by: ", author, " has: ", len(chunks), "chunks.")
                    
                    for i in range(len(chunks)):
                        texts.append(chunks[i])
                        labels.append(author_no)
        
        # # ---------------------------------------------------------------------------- #
        print(len(texts), len(labels))
        print(texts[50], labels[50])
        
        print("\nDone processing, starting tokenizer") 
        # tokenizer = Tokenizer()
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        
        # Toy example to test that the tokenizer worked
        # sequences = tokenizer.texts_to_sequences(["Mr. Lupin, I, Sherlock Holmes, think you stole from Mr. Poirot"]); print(sequences)
        # print(tokenizer.sequences_to_texts([[106, 60, 6, 495, 98, 110, 12, 2372, 38, 106, 182]])); exit()
        
        word_index = tokenizer.word_index; #print(word_index['watson']); exit()
        print('Found %s unique tokens.' %len(word_index))
        print('\nLongest sequence: ', max(len(x) for x in sequences))
        data = pad_sequences(sequences, maxlen = maxlen)

        labels = np.asarray(labels)

        print('shape of data tensor:' , data.shape)
        print('shape of label tensor: ', labels.shape)

        # exit()
        # ---------------------------------------------------------------------------- #
        # Shuffle the text chunks
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)

        data = data[indices]
        labels = labels[indices]
        training_samples = int(data.shape[0]*training_split) #; print("\n Train/ Val split: ", training_split)
        val_samples = int(data.shape[0]*(val_split+training_split))
        
        x_train = data[:training_samples]; print("\n Train x shape: ", x_train.shape)
        y_train = labels[:training_samples]

        x_val = data[training_samples: val_samples]; print("\n Val x shape: ", x_val.shape)
        y_val = labels[training_samples: val_samples]
        
        x_test = data[val_samples:]; print("\n Test x shape: ", x_test.shape)
        y_test = labels[val_samples:]

        np.save(processed_path+'x_train.npy', x_train)
        np.save(processed_path+'y_train.npy', y_train)
        np.save(processed_path+'x_val.npy', x_val)
        np.save(processed_path+'y_val.npy', y_val)
        np.save(processed_path+'x_test_CORPUS.npy', x_test)
        np.save(processed_path+'y_test_CORPUS.npy', y_test)  
             
        # ---------------------------------------------------------------------------- #
        for test_type in ['_UNIVERSE', '_AUTHOR']:
            raw_test_dir = test_books_path+test_type
            labels = []
            texts = []

            for author_no, author in enumerate(author_list):
                dir_name = os.path.join(raw_test_dir, author)
                if os.path.exists(dir_name) is not True:
                    print(dir_name)
                    print("Path does not exist")
                    pass
                else:
                    print("\nProcessing test books in :", dir_name, " for ", test_type)
                    for index, fname in enumerate(os.listdir(dir_name)):
                        if fname[-4:] == '.txt':
                            f = open(os.path.join(dir_name, fname))        
                            raw = f.read()
                            f.close()
                        try:
                            start = re.search(r"\*\*\*.*START OF .*\*\*\*", raw).end()
                            stop = re.search(r"\*\*\*.*END OF .*\*\*\*", raw).start()    
                            content = raw[start:stop]
                        except AttributeError:
                            print("Check raw file for", fname)
                            content = raw
                            
                        chunks = chunker(content, chunk_size)  
                        print(fname, "written by: ", author, " has: ", len(chunks), "chunks.")
                        
                        for i in range(len(chunks)):
                            texts.append(chunks[i])
                            labels.append(author_no)
            
            print("\nDone processing, starting tokenizer") 
            sequences = tokenizer.texts_to_sequences(texts)
            print('\nLongest sequence: ', max(len(x) for x in sequences))

            # Pad the test data
            data = pad_sequences(sequences, maxlen=maxlen); print("Test x: ", x_test.shape)
            labels = np.asarray(labels); print("Tesy y: ", y_test.shape)
            
            # Shuffle so that the NN doesn't just cluster these sequences
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)

            x_test = data[indices]
            y_test = labels[indices]
            np.save(processed_path+'x_test'+test_type+'.npy', x_test)
            np.save(processed_path+'y_test'+test_type+'.npy', y_test)

if __name__ == "__main__":
    main()
 