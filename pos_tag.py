import nltk, re, pprint
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pprint, time
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
from collections import Counter
import numpy

path = "/Users/raghebal-ghezi/Github/English Profile/glove.6B.50d.txt" #0.70
# path = 'gn2v.glove.bin' #0.9397245805042038 // w2v_model.bin 0.86
# path = '/Users/raghebal-ghezi/Downloads/deps.contexts' #// 0.13
# path = '/Users/raghebal-ghezi/Downloads/syngcn_embeddings.txt' #0.86 


word2vec = {}

np.random.seed(1234)

with open(path, 'r') as file:
    for line in file:
        row = line.strip().split(' ')
        word2vec[row[0]] = np.float32(row[1:])

def feature(w):
    # vect = {}
    if w in word2vec:
        return word2vec[w]
    else:
        return np.random.rand(50)


tagged_sentence = nltk.corpus.treebank.tagged_sents(tagset='universal')

X = np.concatenate([[feature(pair[0]) for pair in sent] for sent in tagged_sentence])

y = np.concatenate([[pair[1] for pair in sent] for sent in tagged_sentence]).ravel().tolist()

# print(len(X))
# print(len(y))

# print(X[0],y[0])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1234, shuffle=True)

clf = svm.LinearSVC(max_iter=1e4)

clf.fit(X_train, y_train)

y_hat = clf.predict(X_test)

print(classification_report(y_test, y_hat))

