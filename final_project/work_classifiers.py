
def gaussian_classificator(features_train, labels_train):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf

def tree_classificator(features_train, labels_train, **kwargs):

    from sklearn import tree
    clf = tree.DecisionTreeClassifier(**kwargs)
    clf.fit(features_train, labels_train)
    return clf
