import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl



features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="linear")
clf.fit(features_train, labels_train)  
pred = clf.predict(features_test)
print(accuracy_score(pred, labels_test))

