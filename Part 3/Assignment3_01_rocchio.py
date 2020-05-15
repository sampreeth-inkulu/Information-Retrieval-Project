# python Assignment3_<GROUP_NO>_rocchio.py <path to the en_BDNews24 folder>
# <path_to_​ model_queries_<GROUP_NO>.pth> ​ <path_to_​ gold_standard_ranked_list.csv >
# <path_to_​ Assignment2_<GROUP_NO>_ranked_list_A.csv>
import sys
import os
import re
import csv
import string
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Global variables
stop_words = set(stopwords.words('english'))

data_path = ""

num_docs = 89286
vocab = []
vocab_size = 0
index = {}
inv_vocab = {}

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

def top_20_docs(gold_ranked_list_path, retr_ranked_list_path):

    # Reading Gold standard ranked list
    gold_relevance = {}
    with open(gold_ranked_list_path, 'r') as in_file:
        reader = csv.reader(in_file)
        in_file.readline()
        for row in reader:
            query_id = row[0]
            if query_id in gold_relevance:
                gold_relevance[query_id][row[1]] = int(row[2])
            else:
                gold_relevance[query_id] = {}
                gold_relevance[query_id][row[1]] = int(row[2])
    
    # Reading retrieved rank list
    retr_rank_list = {}
    with open(retr_ranked_list_path, 'r') as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            query_id = row[0]
            if query_id in retr_rank_list:
                retr_rank_list[query_id].append(row[1])
            else:
                retr_rank_list[query_id] = []
                retr_rank_list[query_id].append(row[1])
    
    # Obtaining relevant and non relevant docs
    rel_docs = {}
    non_rel_docs = {}
    for query_id in retr_rank_list:
        rel_docs[query_id] = []
        non_rel_docs[query_id] = []

        query_results = retr_rank_list[query_id][:20]

        if query_id in gold_relevance:
            relevance_scores = gold_relevance[query_id]
            for result in query_results:
                if result in relevance_scores:
                    # Is relevance score of the doc equal to 2?
                    if relevance_scores[result] == 2:
                        rel_docs[query_id].append(result)
                    else:
                        non_rel_docs[query_id].append(result)
                else:
                    non_rel_docs[query_id].append(result)
    
    return rel_docs, non_rel_docs



def obtain_query_vector(query_text):
    query_text_freq = Counter(query_text.split())

    query_vector = np.zeros(vocab_size)  

    for term in query_text_freq:
        try:
            i = inv_vocab[term]
            df = index[term][0]

            tf = query_text_freq[term]
            tf = 1 + np.log(tf)
            idf = np.log(num_docs/ df)

            query_vector[i] = tf * idf

        except:
            print("query vector", sys.exc_info())  
    
    # for i in range(vocab_size):
    #     term = vocab[i]
    #     df = index[term][0]

    #     if term in query_text_freq:

    #         tf = query_text_freq[term]
    #         tf = 1 + np.log(tf)
    #         idf = np.log(num_docs/ df)

    #         query_vector[i] = tf * idf
        
    query_vector = query_vector/np.linalg.norm(query_vector)
    return query_vector

def obtain_document_vector(doc_text):
    
    words = word_tokenize(doc_text)
    words = words[3:-3]
    stop_words.add("'s")
    stop_words.add("'t")
            
    lemmatizer = WordNetLemmatizer()

    tokens = [(lemmatizer.lemmatize(w)).lower() for w in words if w not in stop_words and w not in string.punctuation]
    freq = Counter(tokens)

    doc_vector = np.zeros(vocab_size)

    for term in freq:
        try:
            i = inv_vocab[term]
            tf = freq[term]
            doc_vector[i] = 1 + np.log(tf)
        except:
            print("document vector", sys.exc_info())    
    
    # for i in range(vocab_size):
    #     term = vocab[i]

    #     if term in freq:
    #         tf = freq[term]
    #         doc_vector[i] = 1 + np.log(tf)

    doc_vector = doc_vector/np.linalg.norm(doc_vector)
    return doc_vector

def get_doc_path(doc):
    dir = doc.split('.')[2]
    return os.path.join(data_path, dir, doc)

def avg_document_vectors(doc_list):
    sum_vector = np.zeros(vocab_size)
    if len(doc_list) == 0:
        return sum_vector

    for doc in doc_list:
        try:
            with open(get_doc_path(doc), 'r') as f:
                text = re.findall('<TEXT>.+</TEXT>', f.read(), flags=re.DOTALL)       
        except:
            print("sum_document_vectors", sys.exc_info())
            continue
        sum_vector = sum_vector + obtain_document_vector(text[0])
    
    return sum_vector/len(doc_list)


# def update_query_vector(original_query_vector, rel_docs, non_rel_docs, parameters):
#     avg_rel = avg_document_vectors(rel_docs)
#     avg_non_rel = avg_document_vectors(non_rel_docs)

    
if __name__ == "__main__":

    if len(sys.argv) < 5:
        sys.exit("No. of arguments mismatch\nUsage: python Assignment3_01_rocchio.py \
<path to the en_BDNews24 folder> <path_to_​ model_queries_<GROUP_NO>.pth> ​<path_to_​gold_standard_ranked_list.csv> \
<path_to_​ Assignment2_<GROUP_NO>_ranked_list_A.csv>")

    data_path = sys.argv[1]
    indexer_path = sys.argv[2]
    gold_ranked_list_path = sys.argv[3]
    retr_ranked_list_path = sys.argv[4]

    if len(sys.argv) >= 5:
        queries_path = sys.argv[5]
    else:
        queries_path = "queries_01.txt"

    # Loading index
    with open(indexer_path, 'rb') as file:
        index = pickle.load(file)
    
    vocab = list(index.keys())
    vocab_size = len(vocab)
    for i in range(vocab_size):
        inv_vocab[vocab[i]] = i

    rel_docs, non_rel_docs = top_20_docs(gold_ranked_list_path, retr_ranked_list_path)

    query_vectors_rf_wt1 = {}
    query_vectors_rf_wt2 = {}
    query_vectors_rf_wt3 = {}
    query_vectors_psrf_wt1 = {}
    query_vectors_psrf_wt2 = query_vectors_psrf_wt1
    query_vectors_psrf_wt3 = query_vectors_rf_wt3
    scores_rf_wt1 = {}
    scores_rf_wt2 = {}
    scores_rf_wt3 = {}
    scores_psrf_wt1 = {}
    scores_psrf_wt2 = scores_psrf_wt1
    scores_psrf_wt3 = scores_rf_wt3

    parameters = []

    with open(queries_path,'r') as queries:
        
        for query_line in queries.readlines():            
            if (len(query_line) < 2):
                continue

            query_id, query_text = query_line.split(',')
            query_id = int(query_id)

            original_query_vector = obtain_query_vector(query_text)
            avg_rel = avg_document_vectors(rel_docs[query_id])
            avg_non_rel = avg_document_vectors(non_rel_docs[query_id])
            
            query_vectors_rf_wt1[query_id] =       original_query_vector +       avg_rel - 0.5 * avg_non_rel
            query_vectors_rf_wt2[query_id] = 0.5 * original_query_vector + 0.5 * avg_rel - 0.5 * avg_non_rel
            query_vectors_rf_wt3[query_id] =       original_query_vector + 0.5 * avg_rel

            query_vectors_psrf_wt1[query_id] = original_query_vector + avg_rel

            scores_rf_wt1[query_id] = []
            scores_rf_wt2[query_id] = []
            scores_rf_wt3[query_id] = []
            scores_psrf_wt1[query_id] = []

    # Obtaining scores by obtaining all document vectors
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

    for dir in dirs:

        try:
           os.chdir(dir)
        except:
            print("cd failed", sys.exc_info())
            continue
        
        try:
            files = os.listdir()
        except:
            print("ls failed", sys.exc_info())
            continue

        for file in files:
            
            try:
                with open(file, 'r') as f:
                    text = re.findall('<TEXT>.+</TEXT>', f.read(), flags=re.DOTALL)       
            except:
                print("e", sys.exc_info())
                continue
            
            doc_vector = obtain_document_vector(text[0])
                
            for query_id in scores_rf_wt1:
                add_score(scores_rf_wt1[query_id], file, np.dot(query_vectors_rf_wt1[query_id], doc_vector))
            for query_id in scores_rf_wt2:
                add_score(scores_rf_wt2[query_id], file, np.dot(query_vectors_rf_wt2[query_id], doc_vector))
            for query_id in scores_rf_wt3:
                add_score(scores_rf_wt3[query_id], file, np.dot(query_vectors_rf_wt3[query_id], doc_vector))
            
            for query_id in scores_psrf_wt1:
                add_score(scores_psrf_wt1[query_id], file, np.dot(query_vectors_psrf_wt1[query_id], doc_vector))
            
            # if c > 10:
            #     break
        
        try:
            os.chdir('..')
        except:
            print(".. from", dir, "failed", sys.exc_info())
    
    write_output(scores_rf_wt1, "a3_rf_1.csv")
    write_output(scores_rf_wt2, "a3_rf_2.csv")
    write_output(scores_rf_wt3, "a3_rf_3.csv")

    write_output(scores_psrf_wt1, "a3_psrf_1.csv")


