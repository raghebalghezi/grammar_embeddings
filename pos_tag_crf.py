import nltk, re, pprint
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pprint, time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
from collections import Counter
import numpy

# path = "/Users/raghebal-ghezi/Github/English Profile/glove.6B.50d.txt" #0.8922054939514615
path = 'gn2v.glove.bin' #0.9397245805042038


word2vec = {}
with open(path, 'r') as file:
    for line in file:
        row = line.strip().split(' ')
        word2vec[row[0]] = row[1:]

def feature(w):
    vect = {}
    if w in word2vec:
        for idx, dim in enumerate(word2vec[w]):
            vect['Dim{}'.format(idx+1)] = dim
    else:
        for idx, dim in enumerate(list(np.random.rand(50))):
            vect['Dim{}'.format(idx+1)] = dim
    return vect

# print(feature('nice'))



def prepareData(sett):
    X,y = [],[]
    for sent in sett:
        y.append([pair[1] for pair in sent])
        X.append([feature(pair[0]) for pair in sent])
    return X,y


# tagged_sentence = nltk.corpus.treebank.tagged_sents(tagset='universal')
train_set, test_set = train_test_split(tagged_sentence,test_size=0.3,random_state=1234)


X_train,y_train=prepareData(train_set)
X_test,y_test=prepareData(test_set)



crf = CRF(
    algorithm='lbfgs',
    c1=0.01,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

y_pred=crf.predict(X_test)

f1 = metrics.flat_f1_score(y_test, y_pred,average='weighted',labels=crf.classes_)

print(metrics.flat_classification_report(
    y_test, y_pred, labels=crf.classes_, digits=3
))
