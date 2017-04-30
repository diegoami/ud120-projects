#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from  sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import  train_test_split

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
print(data)
labels, features = targetFeatureSplit(data)
print(labels)
print(features)


### it's all yours from here forward!  
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,random_state=42)


clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print(accuracy_score(pred, labels_test))

zero_list = [0]*29
print(accuracy_score(zero_list, labels_test))
comp_matrix = zip(pred,labels_test)
true_pos = [(x,y) for (x,y) in comp_matrix if x == 1 and y == 1]
print(len(true_pos))

print(precision_score(pred, labels_test))
print(recall_score(pred, labels_test))