#!/usr/bin/python

import sys
import traceback
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from work_tester import dump_classifier_and_data,test_classifier

from work_classifiers import *
from work_data_proc import *


target_feature = ['poi']
interesting_features = ['exercised_stock_options','total_stock_value','total_payments','restricted_stock', 'from_poi_to_this_person', 'other','deferred_income', 'long_term_incentive', 'bonus','expenses','salary']



def test_feature(features_list,data_dict, classificator_method, classifier_args):
    print("test_feature wih features %s " % ' '.join(features_list))
    my_dataset = data_dict
    data = featureFormat(data_dict, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    classificator = classificator_method(features, labels, **classifier_args)

    return test_classifier(classificator, my_dataset, features_list)


def get_range_of_feature(data_dict, feature):
    pass

def calculate_accuracy_map(data_dict):
    gauss_call = {"method": gaussian_classificator, "args": {}}
    tree_call = {"method": tree_classificator, "args": dict({"min_samples_split": 2})}
    classifiers = [tree_call, gauss_call]
    outliers_cuts = [0,2,5,10]
    accuracy_map = {}
    for classifier in classifiers:
        for feature in interesting_features:
            for outliers_cut in outliers_cuts:
                pdata_dict = filter_outliers(data_dict,feature, outliers_cut)
                feature_investigated = target_feature + [feature]
                try:
                    result = test_feature(features_list=feature_investigated, data_dict=pdata_dict,
                                          classificator_method=classifier["method"], classifier_args=classifier["args"])
                    accuracy_map[(feature, classifier["method"].__name__,outliers_cut)] = result['accuracy']
                except:
                    traceback.print_exc()
                    print("Skipping feature %s" % feature)
    return accuracy_map

data_dict = load_data_set()

accuracy_map = calculate_accuracy_map(data_dict)

print(accuracy_map )
print sorted(accuracy_map .items(), key=lambda x: x[1], reverse=True)
