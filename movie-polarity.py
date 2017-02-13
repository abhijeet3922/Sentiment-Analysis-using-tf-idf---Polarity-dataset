# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:00:44 2017

@author: Abhijeet Singh
"""

import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold


def make_Dictionary(root_dir):
    emails_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]    
    all_words = []       
    for emails_dir in emails_dirs:
        emails = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
        for mail in emails:
            with open(mail) as m:
                for line in m:
                    words = line.split()
                    all_words += words
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(4000)
    
    np.save('dict_movie.npy',dictionary) 
    
    return dictionary
    
def extract_features(root_dir): 
    emails_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]  
    docID = 0
    features_matrix = np.zeros((2000,4000)) 
    for emails_dir in emails_dirs:
        emails = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
        for mail in emails:
            with open(mail) as m:
                all_words = []
                for line in m:
                    words = line.split()
                    all_words += words
                for word in all_words:
                  wordID = 0
                  for i,d in enumerate(dictionary):
                    if d[0] == word:
                      wordID = i
                      features_matrix[docID,wordID] = all_words.count(word)
            docID = docID + 1                
    return features_matrix
    
#Create a dictionary of words with its frequency

root_dir = 'txt_sentoken'
dictionary = make_Dictionary(root_dir)


#Prepare feature vectors per training mail and its labels

features_matrix = extract_features(root_dir)
labels = np.zeros(2000);
labels[0:1000]=0;
labels[1000:2000]=1;
np.save('movie_features_matrix.npy',features_matrix)
np.save('movie_labels.npy',labels)


#features_matrix = np.load('movie_features_matrix.npy');
#labels = np.load('movie_labels.npy');

## Training models and its variants
kf = StratifiedKFold(n_splits=10)
totalsvm = 0
totalNB = 0
totalMatSvm = np.zeros((2,2));
totalMatNB = np.zeros((2,2));

for train_index, test_index in kf.split(features_matrix,labels):
    
    X_train = [features_matrix[i] for i in train_index]
    X_test = [features_matrix[i] for i in test_index]
    
    y_train, y_test = labels[train_index], labels[test_index]
    model1 = LinearSVC()
    model2 = MultinomialNB()
    
    model1.fit(X_train,y_train)
    model2.fit(X_train,y_train)
    
    result1 = model1.predict(X_test)
    result2 = model2.predict(X_test)
    
    totalMatSvm = totalMatSvm + confusion_matrix(y_test, result1, labels=[0,1])
    totalMatNB = totalMatNB + confusion_matrix(y_test, result2, labels=[0,1])
    
    totalsvm = totalsvm+sum(y_test==result1)
    totalNB = totalNB+sum(y_test==result2)

print "Confusion matrix for SVM\n",totalMatSvm    
print "True positives - ", totalsvm
print "Confusion matrix for Naive Bayes classifier\n",totalMatNB
print "True positives - ",totalNB
