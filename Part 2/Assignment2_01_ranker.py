import numpy as np
import sys
import os
import csv
import pickle
import operator
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

#Total number of files = 89286 (from Asgn 1)
N = 89286

def add_score(scoring_list, docId, score):
    n = len(scoring_list)
    if (n < 50):
        scoring_list.append([docId, score])
    else:
        min_index = 0
        for i in range(1, n):
            if scoring_list[i][1] < scoring_list[min_index][1]:
                min_index = i
        if scoring_list[min_index][1] < score:
            scoring_list[min_index][0] = docId
            scoring_list[min_index][1] = score

def write_output(scoring, file_name):

    with open(file_name, 'w') as output_file:
        writer = csv.writer(output_file)
        for query_id in scoring:
            results = sorted(scoring[query_id], key=operator.itemgetter(1), reverse=True)
            for result in results:
                writer.writerow([query_id, result[0]])
    
if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        sys.exit("No. of arguments mismatch\nUsage: python Assignment2_01_ranker.py <path to the en_BDNews24 folder> <path to​ model_queries_01.pth>")
    
    data_path = sys.argv[1]
    indexer_path = sys.argv[2]
    if len(sys.argv) >= 4:
        queries_path = sys.argv[3]
    else:
        queries_path = "queries_01.txt"
    
    stop_words = set(stopwords.words('english'))

    # Loading index
    with open(indexer_path, 'rb') as file:
        index = pickle.load(file)
    
    vocab = list(index.keys())
    V = len(vocab)

    print("q1")

    # Query vectors
    q_wt_1 = {}
    q_wt_2 = {}
    q_wt_3 = {}

    scoring1 = {}
    scoring2 = {}
    scoring3 = {}

    with open(queries_path,'r') as queries:
        
        for query_line in queries.readlines():
            
            if (len(query_line) < 2):
                continue

            # print("q1.5")

            query_id, query_text = query_line.split(',')
            query_id = int(query_id)
            query_text_freq = Counter(query_text.split())
            tf_avg = sum(query_text_freq.values())/len(query_text_freq)
            max_tf = max(query_text_freq.values())

            q_wt_1[query_id] = np.array([0]*V)
            q_wt_2[query_id] = np.array([0]*V)
            q_wt_3[query_id] = np.array([0]*V)
            
            # tf idf
            for i in range(V):
                term = vocab[i]
                df = index[term][0]

                if term in query_text_freq:

                    tf = query_text_freq[term]
                    tf1 = 1 + np.log(tf)
                    idf = np.log(N/ df)

                    q_wt_1[query_id][i] = tf1 * idf
                    
                    tf2 = (1 + np.log(tf))/(1 + np.log(tf_avg))
                    idf = max(0, np.log((N- df)/df))

                    q_wt_2[query_id][i] = tf2 * idf

                    tf3 = 0.5 + (0.5*tf)/max_tf

                    q_wt_3[query_id][i] = tf3 * idf
                # else:
                    # idf = max(0, np.log((N- df)/df))
                    # tf3 = 0.5
                    # q_wt_3[query_id][i] = tf3 * idf

            
            # normalize
            q_wt_1[query_id] = q_wt_1[query_id]/np.linalg.norm(q_wt_1[query_id])
            q_wt_2[query_id] = q_wt_2[query_id]/np.linalg.norm(q_wt_2[query_id])
            q_wt_3[query_id] = q_wt_3[query_id]/np.linalg.norm(q_wt_3[query_id])

            scoring1[query_id] = []
            scoring2[query_id] = []
            scoring3[query_id] = []

    print("q2")
    try:
        cwd = os.getcwd()
    except:
        pass

    try:
        os.chdir(data_path)
    except:
        sys.exit("Couldn't change directory to", data_path)

    try:
        dirs = os.listdir()
    except:
        sys.exit("Couldn't get list of files in ", data_path)
    c = 0
    for dir in dirs:

        try:
           os.chdir(dir)
        except:
            print("a", sys.exc_info())
            continue
        
        try:
            files = os.listdir()
        except:
            print("b", sys.exc_info())
            continue
        for file in files:
            # print(file)
            c += 1
            try:
                with open(file, 'r') as f:
                    text = re.findall('<TEXT>.+</TEXT>', f.read(), flags=re.DOTALL)       
            except:
                print("e", sys.exc_info())
                continue

            words = word_tokenize(text[0])

            words = words[3:-3]
            stop_words.add("'s")
            stop_words.add("'t")
            
            lemmatizer = WordNetLemmatizer()

            tokens = [(lemmatizer.lemmatize(w)).lower() for w in words if w not in stop_words and w not in string.punctuation]
            freq = Counter(tokens)
            tf_avg = sum(freq.values())/len(freq)
            max_tf = max(freq.values())

            doc_wt1 = np.array([0]*V)
            doc_wt2 = np.array([0]*V)
            doc_wt3 = np.array([0]*V)
            # print("q2.3")
            for i in range(V):
                term = vocab[i]
                if term in freq:

                    tf = freq[term]

                    doc_wt1[i] = 1 + np.log(tf)                    
                    doc_wt2[i] = (1 + np.log(tf))/(1 + np.log(tf_avg))
                    doc_wt3[i] = 0.5 + (0.5*tf)/max_tf
                
            for query_id in scoring1:
                # print("q2.5")
                add_score(scoring1[query_id], file, np.dot(q_wt_1[query_id], doc_wt1))
                add_score(scoring2[query_id], file, np.dot(q_wt_2[query_id], doc_wt2))
                add_score(scoring3[query_id], file, np.dot(q_wt_3[query_id], doc_wt3))
            
            # if c > 10:
            #     break
        
        try:
            os.chdir('..')
        except:
            print(".. from", dir, "failed", sys.exc_info())

        # if c > 10:
        #     break
    
    print("scoring1", scoring1)
    print("q3")
    try:
        os.chdir(cwd)
    except:
        cwd = os.getcwd()

    write_output(scoring1, "Assignment2_01_ranked_list_A.csv")
    write_output(scoring2, "Assignment2_01_ranked_list_B.csv")
    write_output(scoring3, "Assignment2_01_ranked_list_C.csv")
    print("Done!")