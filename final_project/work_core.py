
from feature_format import featureFormat, targetFeatureSplit
from work_tester import dump_classifier_and_data,test_classifier

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