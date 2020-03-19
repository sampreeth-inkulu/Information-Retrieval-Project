import os
import re
import sys
import pickle
import string
import time
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        sys.exit("No. of arguments mismatch\nUsage: python Assignment1_01_indexer.py <path to the en_BDNews24 folder>")
    
    path = sys.argv[1]

    start = time.time()
    index = {}
    stop_words = set(stopwords.words('english'))

    try:
        cwd = os.getcwd()
    except:
        pass

    try:
        os.chdir(path)
    except:
        sys.exit("Couldn't change directory to", path)

    try:
        dirs = os.listdir()
    except:
        sys.exit("Couldn't get list of files in ", path)

    # Reading files
    for dir in sorted(dirs):

        try:
           os.chdir(dir)
        except:
            continue
        
        try:
            files = os.listdir()
        except:
            continue

        for file in sorted(files):

            try:
                with open(file, 'r') as f:
                    text = re.findall('<TEXT>.+</TEXT>', f.read(), flags=re.DOTALL)
            except:
                continue
            
            # Tokenization
            words = word_tokenize(text[0])

            words = words[3:-3]
            stop_words.add("'s")
            stop_words.add("'t")
            
            lemmatizer = WordNetLemmatizer()

            tokens = [(lemmatizer.lemmatize(w)).lower() for w in words if w not in stop_words and w not in string.punctuation]

            # Index
            for token in set(tokens):
                
                if token in index:
                    index[token][1].append(file)
                else:
                    index[token] = [0, [file]]

        try:
            os.chdir('..')
        except:
            print(".. from", dir, "failed", sys.exc_info())
    
    try:
        os.chdir(cwd)
    except:
        cwd = os.getcwd()

    # Document frequency
    for term in index:
        index[term][0] = len(index[term][1])
    
    # Saving the index object
    with open('model_queries_01.pth', 'wb') as file:
        pickle.dump(index, file)

    end = time.time()
    print("Vocabulary size =", len(index))
    print("model_queries_01.pth saved in directory", cwd)
    print("Time taken to index =", end - start, "seconds")