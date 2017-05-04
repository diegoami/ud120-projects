import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from work_tester import dump_classifier_and_data,test_classifier
from work_data_proc import *
from work_data_proc import remove_useless
from sklearn.preprocessing import MinMaxScaler
from work_parameters import *
import numpy as np
from sklearn.metrics import accuracy_score



def create_classificator(features_list,data_dict, classificator_method, classifier_args):
    print("test_feature wih features %s " % ' '.join(features_list))
    my_dataset = data_dict
    features_train, labels_train = retrieve_data_for_classifier(data_dict, features_list)
    classificator = classificator_method(features_train, labels_train, **classifier_args)
    return classificator


def retrieve_data_for_classifier(data_dict, features_list):
    pdata_dict = remove_useless(data_dict)
    data = featureFormat(pdata_dict, features_list, sort_keys=True)
    print(type(data))
    labels_aslist, features_aslist = targetFeatureSplit(data)
    labels = data[:,:1]
    features = data[:, 1:]
    print(labels.shape)
    print(features.shape)
    print(len(labels_aslist))
    print(len(features_aslist),len(features_aslist[0]) )

    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
       train_test_split(features, labels, test_size=0.3, random_state=42)
    print(type(features_train), type(labels_train))
    #scaler = MinMaxScaler()
    #rescaled_features = scaler.fit_transform(features_train)
    print(features_train.shape)
    print(labels_train.shape)
    return features_train, labels_train
    #return rescaled_features, labels_train
    #return  features_train, labels_train

def get_scaled_nparray(features):
    np_max_feature = np.max(features,axis=0)
    np_min_feature = np.min(features, axis=0)
    return np_max_feature, np_min_feature

if __name__ == "__main__":
    data_dict = load_data_set()
    pdata_dict = remove_useless(data_dict)
    data = featureFormat(pdata_dict, all_features, sort_keys=True)
    print(type(data))
    print(data.shape)
    only_features = data[:,1:]
    print(type(only_features ))
    print(only_features .shape)

    #scaled_features, labels_train = retrieve_data_for_classifier( pdata_dict,ana_features)
    #print(type(scaled_features))
    #scpna = get_scaled_nparray(features)
    #print(scpna)
