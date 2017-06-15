def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

     
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    
    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)
   
    correct = 0
    for i in range(0,len(pred)):
        if pred[i] == labels_test[i]:
            correct = correct+1
    accuracy = correct/len(pred)
    return accuracy