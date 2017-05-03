import pickle
import numpy as np
from work_parameters import *

def remove_useless(data_dict):
    data_dict = {k:v for k,v in data_dict.items() if any(v[cf] != 'NaN' for cf in crux_feat)}
    return data_dict

def load_data_set():
    with open("data/final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
        data_dict.pop("TOTAL")
        return data_dict

def conv_float(x):
    if type(x) == int or type(x) == float:
        return x
    try:
        return float(x)
    except:
        return x

def sort_by_feature(data_dict, feature):
    sdata_dict = sorted(data_dict.items(),key=lambda x: conv_float(x[1][feature]) )
    return sdata_dict

def remove_NaN(data_dict, feature):
    gdata_dict= {k:v for k, v in data_dict.items() if v[feature] !='NaN' }
    return gdata_dict

def remove_outliers(data_list, percentage):
    ldl = len(data_list)
    fpert = float(percentage)
    lbeg = int(ldl*fpert/100)
    nooutl_list = data_list[:ldl-lbeg]
    return nooutl_list

def get_workable_list(data_dict, feature, percentage=None):
    data_dict = remove_NaN(data_dict, feature)
    data_list = sort_by_feature(data_dict, feature)
    if (percentage):
        data_list = remove_outliers(data_list, percentage)
    return data_list

def filter_dictionary(data_dict, feature, percentage):
    data_list = get_workable_list(data_dict, feature, percentage)
    noutl_dict = {k:v for (k,v) in data_list  }
    return noutl_dict

def average_feature(data_dict, feature, percentage=0):
    data_list = get_workable_list(data_dict, feature, percentage)
    sum_feature = sum([conv_float(x[1][feature]) for x in data_list])
    return sum_feature / len(data_list)

def fill_NaN_with_average(data_dict, feature, percentage=0):
    average = average_feature(data_dict, feature, percentage)
    for k, v in data_dict.items():
        v[feature] = average if v[feature] == 'NaN' else v[feature]
    return data_dict

def range_feature(data_dict, feature, percentage=0):
    data_list = get_workable_list(data_dict, feature, percentage)
    max_feature = max(data_list,key=lambda x: conv_float(x[1][feature])  )
    min_feature = min(data_list,key=lambda x: conv_float(x[1][feature] ) )
    return int(min_feature[1][feature]), int(max_feature[1][feature])

def scale_data(data_dict, feature, percentage=0):
    min_feat, max_feat = range_feature(data_dict, feature, percentage)
    for k, v in data_dict:
        v[feature] = (v[feature] - min_feat)/max_feat

def test_drive():
    data_dict = load_data_set()
    for feature in features:
        data_dict = fill_NaN_with_average(data_dict ,feature)
        for perc in outliers_cuts :
            print(feature,  perc, range_feature(data_dict,feature,perc))

if __name__ == "__main__":
    test_drive()