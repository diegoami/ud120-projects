#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

for cvalue in [10000.0]:
    print('CValue='+str(cvalue))
    clf = SVC(kernel='rbf', C=cvalue)
    t0 = time()
    clf.fit(features_train, labels_train)
    print( "training time:", round(time()-t0, 3), "s")
    ### use the trained classifier to predict labels for the test features
    t0 = time()
    pred = clf.predict(features_test)
    print "prediction time:", round(time()-t0, 3), "s"
    print(accuracy_score(pred, labels_test))
    print  (pred[10], pred[26], pred[50])
    print(len([x for x in pred if x == 1]))
    

#########################################################
### your code goes here ###

#########################################################


