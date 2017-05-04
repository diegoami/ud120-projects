
import numpy as np

def gaussian_classificator(features_train, labels_train):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, np.ravel(labels_train))
    return clf

def tree_classificator(features_train, labels_train, **kwargs):

    from sklearn import tree
    clf = tree.DecisionTreeClassifier(**kwargs)
    clf.fit(features_train, np.ravel(labels_train))
    return clf

def svc_classificator(features_train, labels_train):
    from sklearn.svm import SVC

    ### your code goes here!
    clf = SVC()
    clf.fit(features_train, labels_train)
    return clf


gauss_call = {"method": gaussian_classificator, "args": {}}

tree_call = {"method": tree_classificator, "args": dict({"min_samples_split": 2})}
svc_call = {"method": svc_classificator, "args": {}}

classifiers = [  tree_call, gauss_call]