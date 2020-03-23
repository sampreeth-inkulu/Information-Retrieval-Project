import os
import math
import operator
import csv
import sys
import re
import string
import statistics
import time
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

N=89286

def cos_sim(qvec,dvec):
    sum=0
    for i in range(len(qvec)):
        sum+=qvec[i]*dvec[i]
    return sum

#Total number of files = 89286 (from Asgn 1)
if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("No. of arguments mismatch\nUsage: python Assignment2_01_ranker.py <path to Data folder> <path to model>")

    data_path = sys.argv[1]
    model_path = sys.argv[2]
    stop_words = set(stopwords.words('english'))
    # Loading index
    with open(model_path, 'rb') as file:
        index = pickle.load(file)
    
    #DF(term) == index[term][0]
    
    vocab=index.keys()
    doc_wt={}

    try:
        os.chdir(data_path)
    except:
        sys.exit("Couldn't change directory to", data_path)

    try:
        dirs = os.listdir()
    except:
        sys.exit("Couldn't get list of files in ", data_path)

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
            freq=Counter(tokens)
            doc_wt[file]=[]
            norm=0
            for v in vocab:
                if freq[v]!=0:
                    tf=0.5+(0.5*freq[v])/(max(freq.values()))
                    doc_wt[file].append(tf)
                    norm+=tf*tf
                else:
                    doc_wt[file].append(0)
            norm=math.sqrt(norm)
            for i in range(len(vocab)):
                doc_wt[file][i]/=norm
            
            

            
        try:
            os.chdir('..')
        except:
            print(".. from", dir, "failed", sys.exc_info())
    
    try:
        os.chdir(cwd)
    except:
        cwd = os.getcwd()
    os.chdir('..')

    queries=open('queries_01.txt','r')

    q_wt={}
    for line in queries.readlines():
        norm=0
        qid,qtxt=line.split(',')
        qid=int(qid)
        qtxt=Counter(qtxt.split())
        tf_avg=statistics.mean(qtxt.values())
        q_wt[qid]=[]
        for v in vocab:
            if v in qtxt:
                tf=0.5+(0.5*qtxt[v])/(max(qtxt.values()))
                df=index[v][0]
                idf=max(0,math.log((N-df)/df))
                wt=tf*idf
                norm=wt*wt
                q_wt[qid].append(wt)
            else:
                q_wt[qid].append(0)
        norm=math.sqrt(norm)
        if(norm!=0):
            for i in range(len(vocab)):
                q_wt[qid][i]/=norm
    queries.close()
    # print(q_wt)
    #Ranking
    output=open('Assignment2_01_ranked_list_C.csv','w')
    csvwriter = csv.writer(output)
    for qid in q_wt:
        score={}
        for doc in doc_wt:
            score[doc]=cos_sim(q_wt[qid],doc_wt[doc])
        # print(score)
        top_50=sorted(score.items(), key=operator.itemgetter(1),reverse=True)
        # print(top_50)
        top_50=top_50[:50]
        
        for row in top_50:
            # print([qid,row[0]])
            csvwriter.writerow([qid,row[0]])
    output.close()

            
    

