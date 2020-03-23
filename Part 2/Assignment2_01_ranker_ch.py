import os
import math
import operator
import csv
import sys
import re
import statistics
import string
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
    doc_wt_1={}
    doc_wt_2={}
    doc_wt_3={}

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
            tf_avg=statistics.mean(freq.values())
            doc_wt_1[file]=[]
            doc_wt_2[file]=[]
            doc_wt_3[file]=[]
            norm1=0
            norm2=0
            norm3=0
            for v in vocab:
                if freq[v]!=0:
                    tf=1+math.log(freq[v])
                    doc_wt_1[file].append(tf)
                    norm1+=tf*tf
                    tf=(1+math.log(freq[v]))/(1+math.log(tf_avg))
                    doc_wt_2[file].append(tf)
                    norm2+=tf*tf
                    tf=0.5+(0.5*freq[v])/(max(freq.values()))
                    doc_wt_3[file].append(tf)
                    norm3+=tf*tf
                else:
                    doc_wt_1[file].append(0)
                    doc_wt_2[file].append(0)
                    doc_wt_3[file].append(0)
            norm1=math.sqrt(norm1)
            norm2=math.sqrt(norm2)
            norm3=math.sqrt(norm3)
            for i in range(len(vocab)):
                doc_wt_1[file][i]/=norm1
                doc_wt_2[file][i]/=norm2
                doc_wt_3[file][i]/=norm3
            
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

    q_wt_1={}
    q_wt_2={}
    q_wt_3={}
    for line in queries.readlines():
        norm1=0
        norm2=0
        norm3=0
        qid,qtxt=line.split(',')
        qid=int(qid)
        qtxt=Counter(qtxt.split())
        q_wt_1[qid]=[]
        q_wt_2[qid]=[]
        q_wt_3[qid]=[]
        for v in vocab:
            if v in qtxt:
                df=index[v][0]
                tf=1+math.log(qtxt[v])
                idf=math.log(N/df)
                wt=tf*idf
                norm1=wt*wt
                q_wt_1[qid].append(wt)
                tf=(1+math.log(qtxt[v]))/(1+math.log(tf_avg))
                idf=max(0,math.log((N-df)/df))
                wt=tf*idf
                norm2=wt*wt
                q_wt_2[qid].append(wt)
                tf=0.5+(0.5*qtxt[v])/(max(qtxt.values()))
                idf=max(0,math.log((N-df)/df))
                wt=tf*idf
                norm3=wt*wt
                q_wt_3[qid].append(wt)
            else:
                q_wt_1[qid].append(0)
                q_wt_2[qid].append(0)
                q_wt_3[qid].append(0)
        norm1=math.sqrt(norm1)
        norm2=math.sqrt(norm2)
        norm3=math.sqrt(norm3)
        for i in range(len(vocab)):
            q_wt_1[file][i]/=norm1
            q_wt_2[file][i]/=norm2
            q_wt_3[file][i]/=norm3
    queries.close()
    # print(q_wt)
    #Ranking
    output1=open('Assignment2_01_ranked_list_A.csv','w')
    csvwriter1 = csv.writer(output1)
    output2=open('Assignment2_01_ranked_list_B.csv','w')
    csvwriter2 = csv.writer(output2)
    output3=open('Assignment2_01_ranked_list_C.csv','w')
    csvwriter3 = csv.writer(output3)
    for qid in q_wt_1:
        score={}
        for doc in doc_wt_1:
            score[doc]=cos_sim(q_wt_1[qid],doc_wt_1[doc])
        top_50=sorted(score.items(), key=operator.itemgetter(1),reverse=True)
        top_50=top_50[:50]
        for row in top_50:
            csvwriter1.writerow([qid,row[0]])
    for qid in q_wt_2:
        score={}
        for doc in doc_wt_2:
            score[doc]=cos_sim(q_wt_2[qid],doc_wt_2[doc])
        top_50=sorted(score.items(), key=operator.itemgetter(1),reverse=True)
        top_50=top_50[:50]
        for row in top_50:
            csvwriter2.writerow([qid,row[0]])
    for qid in q_wt_3:
        score={}
        for doc in doc_wt_3:
            score[doc]=cos_sim(q_wt_3[qid],doc_wt_3[doc])
        top_50=sorted(score.items(), key=operator.itemgetter(1),reverse=True)
        top_50=top_50[:50]
        for row in top_50:
            csvwriter3.writerow([qid,row[0]])
    output1.close()
    output2.close()
    output3.close()

            
    

