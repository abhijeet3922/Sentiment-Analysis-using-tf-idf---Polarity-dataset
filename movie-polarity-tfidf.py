# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 17:03:30 2017

@author: Abhijeet Singh
"""

import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

def make_Corpus(root_dir):
    polarity_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]    
    corpus = []    
  
    for polarity_dir in polarity_dirs:
        reviews = [os.path.join(polarity_dir,f) for f in os.listdir(polarity_dir)]
        for review in reviews:
            doc_string = "";
            with open(review) as rev:
                for line in rev:
                    doc_string = doc_string + line
            if not corpus:
                corpus = [doc_string]
            else:
                corpus.append(doc_string)
    return corpus

#Create a dictionary of words with its frequency

root_dir = 'txt_sentoken'
corpus = make_Corpus(root_dir)


#Prepare feature vectors per training mail and its labels
labels = np.zeros(2000);
labels[0:1000]=0;
labels[1000:2000]=1; 
      
kf = StratifiedKFold(n_splits=10)

totalsvm = 0
totalNB = 0
totalMatSvm = np.zeros((2,2));
totalMatNB = np.zeros((2,2));

for train_index, test_index in kf.split(corpus,labels):
    
    X_train = [corpus[i] for i in train_index]
    X_test = [corpus[i] for i in test_index]
    
    y_train, y_test = labels[train_index], labels[test_index]
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True,stop_words='english')
    train_corpus_tf_idf = vectorizer.fit_transform(X_train) 
    test_corpus_tf_idf = vectorizer.transform(X_test)
    
    model1 = LinearSVC()
    model2 = MultinomialNB()

    
    model1.fit(train_corpus_tf_idf,y_train)
    model2.fit(train_corpus_tf_idf,y_train)

    
    result1 = model1.predict(test_corpus_tf_idf)
    result2 = model2.predict(test_corpus_tf_idf)
    
    totalMatSvm = totalMatSvm + confusion_matrix(y_test, result1, labels=[0,1])
    totalMatNB = totalMatNB + confusion_matrix(y_test, result2, labels=[0,1])
    
    totalsvm = totalsvm+sum(y_test==result1)
    totalNB = totalNB+sum(y_test==result2)
    
print "Confusion matrix for SVM\n",totalMatSvm    
print "True positives - ", totalsvm
print "Confusion matrix for Naive Bayes classifier\n",totalMatNB
print "True positives - ",totalNB
