#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g", "y"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
#print(data_dict)
dataf = [x for x in data_dict.values() if x["exercised_stock_options"] != 'NaN']
mind = min(dataf  ,key=lambda x: x["exercised_stock_options"] )
maxd = max(dataf ,key=lambda x: x["exercised_stock_options"] )

print(mind['exercised_stock_options'])
print(maxd['exercised_stock_options'])


datag = [x for x in data_dict.values() if x["salary"] != 'NaN']
ming = min(datag  ,key=lambda x: x["salary"] )
maxg = max(datag ,key=lambda x: x["salary"] )

print(ming['salary'])
print(maxg['salary'])



poi, finance_features = targetFeatureSplit( data )

#print(finance_features)
### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
minf2, maxf2 = None,None

for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
    if maxf2 == None:
        maxf2 = f2
    if minf2 == None:
        minf2 = f2
    if f2 > maxf2:
        maxf2 = f2
    if f2 < minf2:
        minf2 = f2

plt.show()

#print(minf2)
#print(maxf2)

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
print(finance_features)
scaler=MinMaxScaler()
rescaled_features = scaler.fit_transform(finance_features)
print(rescaled_features)
km = KMeans(n_clusters=2)
km.fit(rescaled_features)
pred = km.predict(rescaled_features)

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, rescaled_features, poi, mark_poi=False, name="clusters_2_w.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
