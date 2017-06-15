#!/usr/bin/python

""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.
    
    The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """


from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
import numpy as np
import pylab as pl
import random
from sklearn.metrics import accuracy_score

def classifyNB(features_train, labels_train):   
    from sklearn.naive_bayes import GaussianNB
    
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf


def classifySVC(features_train, labels_train, kernelArg,gammaArg='auto',carg=1.0): 
   from sklearn.svm import SVC
   clf = SVC(kernel=kernelArg, gamma=gammaArg,C=carg)
   clf.fit(features_train, labels_train)  
   return clf


def classifyTree(features_train, labels_train,min_samples_split_arg=2): 
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split=min_samples_split_arg)
    clf.fit(features_train, labels_train)  
    return clf



features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


# You will need to complete this function imported from the ClassifyNB script.
# Be sure to change to that code tab to complete this quiz.



### draw the decision boundary with the text points overlaid
def doSVC(features_train, labels_train, features_test, labels_test ):
    for i in range(1,20):
        cargrand = random.random()*10000
        clf = classifySVC(features_train, labels_train, kernelArg='rbf', carg=cargrand)
        filename = "testRBF_"+str(cargrand)+".png"
        prettyPicture(clf, features_test, labels_test,filename)
        output_image(filename, "png", open(filename, "rb").read())


def doTree(features_train, labels_train, features_test, labels_test ):
    max_accuracy, index = 0, 0
    for i in np.arange(5,200,5):
        print('min_samples_split_arg='+str(i))
        clf = classifyTree(features_train, labels_train,min_samples_split_arg=i)
        filename = "testtree/testTree_"+str(i)+".png"
        prettyPicture(clf, features_test, labels_test,filename)
        output_image(filename, "png", open(filename, "rb").read())
        pred = clf.predict(features_test)
        acc = accuracy_score(pred, labels_test)
        print(acc)
        if acc > max_accuracy:
            max_accuracy = acc
            index = i
    print(max_accuracy , index)
        

clf = doTree(features_train, labels_train, features_test, labels_test)
#pred = clf.predict(features_test)
#print(accuracy_score(pred, labels_test))




