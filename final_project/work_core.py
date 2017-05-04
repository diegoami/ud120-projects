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

def get_maxmin_nparray(features):
    np_max_feature = np.max(features,axis=0)
    np_min_feature = np.min(features, axis=0)
    return np_max_feature, np_min_feature

def get_scaled_nparray(features,np_min_feature,np_max_feature ):
    scaled_features = (features-np_min_feature)/(np_max_feature-np_min_feature)
    return scaled_features

def get_count_nonan(features_withNan):
    count_nan = features_withNan


if __name__ == "__main__":
    data_dict = load_data_set()
    pdata_dict = remove_useless(data_dict)
    data = featureFormat(pdata_dict, ana_features, sort_keys=True, remove_NaN=True)
    labels = data[:, :1]
    features = data[:, 1:]
    np.savetxt("csv/features.csv", np.asarray(features ), delimiter=",",fmt='%10.2f')
    max_feature, min_feature = get_maxmin_nparray(features)
    print(max_feature, min_feature )
    scaled_features = get_scaled_nparray(features,min_feature ,max_feature )
    np.savetxt("csv/scaled_features.csv", np.asarray(scaled_features ), delimiter=",",fmt='%2.4f')
    data_with_nan = featureFormat(pdata_dict, ana_features, sort_keys=True, remove_NaN=False)
    labels_with_nan = data_with_nan[:, :1]
    features_with_nan = data_with_nan[:, 1:]

    nan = np.isnan(features_with_nan )
    not_nan = np.logical_not(nan)
    features_with_nan[nan]
    np.savetxt("csv/features_nan.csv", np.asarray(features_with_nan ), delimiter=",",fmt="%10.2f")
    np.savetxt("csv/nan.csv", np.asarray(nan), delimiter=",",fmt="%1.0f")
    np.savetxt("csv/not_nan.csv", np.asarray(not_nan), delimiter=",", fmt="%1.0f")

    col_no_nan = np.count_nonzero(not_nan,axis=1)
   # print(col_no_nan)
    np.savetxt("csv/col_no_nan.csv", np.asarray(col_no_nan ), delimiter=",", fmt="%2.0f")
    imputed_scaled_feature = scaled_features.sum(axis=1)/col_no_nan
    np.savetxt("csv/imputed_scaled_feature.csv", np.asarray(imputed_scaled_feature), delimiter=",", fmt='%2.4f')
    features_imputed = scaled_features.copy()
    #print(features_imputed.shape)

    #print(nan.shape)

    all_scaled_feature = np.stack([imputed_scaled_feature for _  in range(features_imputed.shape[1]) ],axis=1)
   # all_tiled_feature = np.tile(imputed_scaled_feature , ( features_imputed.shape[1],1)).T
   # all_tiled_feature_2 = np.tile(imputed_scaled_feature, ( 1,features_imputed.shape[1]))

    print(all_scaled_feature.shape)
   # print(all_tiled_feature.shape)
   # print(all_tiled_feature_2.shape)

    features_imputed = features_imputed +  all_scaled_feature * nan

    #np.place(features_  imputed , features_imputed[nan], imputed_scaled_feature)

    np.savetxt("csv/features_imputed.csv", np.asarray(features_imputed), delimiter=",", fmt='%2.4f')
    #features_with_nan[nan] = 0.5
    #scaled_features, labels_train = retrieve_data_for_classifier( pdata_dict,ana_features)
    #print(type(scaled_features))
    #scpna = get_scaled_nparray(features)
    #print(scpna)
    """
    print(scaled_features.shape)
    print(features_imputed.shape)
    print(imputed_scaled_feature.shape)
    print(np.ravel(imputed_scaled_feature).shape)
    print(type(imputed_scaled_feature))
    transposed_scaled_feature = imputed_scaled_feature.reshape(1,-1)
    print(imputed_scaled_feature.shape)
    print(transposed_scaled_feature .shape)
    print(imputed_scaled_feature)
    print(transposed_scaled_feature )

    npdot = np.dot(transposed_scaled_feature * nan)
    print(npdot)
    """