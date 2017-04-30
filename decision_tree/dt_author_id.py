#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score

def classifyTree(features_train, labels_train,min_samples_split_arg=2): 
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split=min_samples_split_arg)
   # clf.fit(features_train, labels_train)
    return clf
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print("Features in train"+str(len(features_train[0])))



#########################################################
### your code goes here ###


#########################################################
clf =  classifyTree(features_train, labels_train,min_samples_split_arg=40)
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

