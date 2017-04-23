#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
l = float(len(enron_data.keys()))
poic = 0

#feature_james = enron_data['PRENTICE JAMES']
#feature_james = enron_data['COLWELL WESLEY']
feature_james = enron_data['SKILLING JEFFREY K']
print feature_james
for person_key, features_dict in enron_data.iteritems():
    #print len(features_dict.keys() )
    #print person_key
    if features_dict["poi"] :
        poic = poic+1
#print(poic)
    
#mykeys = ["SKILLING JEFFREY K","LAY KENNETH L","FASTOW ANDREW S"]
print(len(enron_data.keys()))
print(len([v for k, v in enron_data.iteritems() if v['salary'] != 'NaN' ]))
print(len([v for k, v in enron_data.iteritems() if v['email_address'] != 'NaN' ]))
atp = len([v for k, v in enron_data.iteritems() if v['total_payments'] == 'NaN' ])
pou = float(len([v for k, v in enron_data.iteritems() if v['poi'] ]))
atpp = len([v for k, v in enron_data.iteritems() if v['total_payments'] !='NaN' and v['poi'] ] )
print atp
print(atp/l)
print pou
print(atpp/pou)
