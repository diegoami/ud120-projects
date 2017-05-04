#!/usr/bin/python

import sys
import traceback
sys.path.append("../tools/")

from work_classifiers import *
from work_data_proc import *
from work_core import *

target_feature = ['poi']

outliers_cuts = [0, 2, 5, 10]
interesting_features = ['exercised_stock_options', 'total_stock_value', 'total_payments', 'restricted_stock', \
                        'from_poi_to_this_person', 'other', 'deferred_income', 'long_term_incentive', 'bonus', \
                        'expenses', 'salary']



def iterate_params(interesting_features, classifiers, outliers_cuts ):
    for classifier in classifiers:  # classifiers:
        for feature in interesting_features:  # interesting_features:
            analyzed_features = [feature]
            for outliers_cut in outliers_cuts :  # outliers_cuts:
                second_feature_list = [] + [[item] for item in interesting_features if item not in analyzed_features ]
                yield {"classifier":classifier, "feature_list" : target_feature +[feature], "outliers_cut" : outliers_cut, "outlier_feature" : feature}


def iterate_params_all(classifiers):
    for classifier in classifiers:  # classifiers:
         yield {"classifier":classifier, "feature_list" : target_feature +ana_features, "outliers_cut" : 100, "outlier_feature" : []}


def get_range_of_feature(data_dict, feature):
    pass

def calculate_accuracy_map(data_dict):
    accuracy_map = {}
    #for _ in iterate_params(interesting_features=interesting_features, classifiers=classifiers, outliers_cuts=outliers_cuts):
    for _ in iterate_params_all(classifiers=classifiers):

        #pdata_dict = filter_dictionary(data_dict, _["outlier_feature"], _["outliers_cut"])
        try:
            pass
            classif = create_classificator(features_list=_["feature_list"], data_dict=data_dict,
                                  classificator_method=_["classifier"]["method"], classifier_args=_["classifier"]["args"])
            result = test_classifier( classif , data_dict, all_features)
            #accuracy = result['accuracy']
            accuracy_map[("|".join(_["feature_list"]),_["classifier"]["method"].__name__,_["outliers_cut"])] = result['accuracy']
        except:
            traceback.print_exc()
            print("Skipping feature_list %s" % "|".join(_["feature_list"]))
    return accuracy_map


if __name__ == "__main__":
    data_dict = load_data_set()
    accuracy_map = calculate_accuracy_map(data_dict)
    print(accuracy_map )
    print sorted(accuracy_map .items(), key=lambda x: x[1], reverse=True)
