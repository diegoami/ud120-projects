#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn import ensemble
from sklearn.metrics import accuracy_score
features_train, labels_train, features_test, labels_test = makeTerrainData()

import numpy

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary


print("Features ="+str(len(features_train[0])))
best_seed, best_min_sample, best_estimators, best_accur =0,0,0,0

def doRandomTree(seed_arg=4, min_samples_split_arg=20,n_estimators_arg=10):
    
    print("Seed_arg="+str(seed_arg))
    print("min_samples_split_arg="+str(min_samples_split_arg))
    print("n_estimators_arg="+str(n_estimators_arg))
    filename = "imgs/test_seed-"+str(seed_arg)+"_minsamples-"+str(min_samples_split_arg)+"_nestimators-"+str(n_estimators_arg)+".png"
    
    clf = ensemble.RandomForestClassifier(bootstrap=False,n_estimators=n_estimators_arg, random_state=seed_arg, min_samples_split=min_samples_split_arg)
    clf.fit(features_train, labels_train)  
    
    try:
        prettyPicture(clf, features_test, labels_test,filename)
    except NameError:
        pass
    
    pred = clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    print("accuracy="+str(acc))
    return acc  

def loop_seed_min_sample_split():
    for seed_arg in range(1,10):
        for min_samples_split_arg in numpy.arange(5,30,5):
            acc = doRandomTree(seed_arg, min_samples_split_arg)
            if acc >  best_accur:
                best_seed, best_min_sample, best_accur = seed_arg, min_samples_split_arg,acc

def loop_seed_min_sample_split_nestimators():
    best_seed, best_min_sample, best_estimators, best_accur =0,0,0,0
    for seed_arg in [4,66]:
        for min_samples_split_arg in range(15,22):
            for n_estimators_arg in range(5,15): 
                acc = doRandomTree(seed_arg, min_samples_split_arg,n_estimators_arg)
                if acc >  best_accur:
                    best_seed, best_min_sample, best_estimators, best_accur = seed_arg, min_samples_split_arg,n_estimators_arg, acc    
    print("best result")            
    print(best_seed, best_min_sample, best_estimators, best_accur )


    
loop_seed_min_sample_split_nestimators()


