#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
from nltk.corpus import stopwords

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = pickle.load(open("your_email_authors.pkl", "r") )
word_data = pickle.load(open("your_word_data.pkl", "r") )


tfid = TfidfTransformer()
tfvect = TfidfVectorizer(stop_words='english',lowercase=True)

"""
print(word_data[152])
new_word_data = []
for sentence in word_data:
    word_list = sentence.split(None, 1)
    filtered_words = [word for word in word_list if word not in stopwords.words('english')]
    new_sentence = ' '.join(filtered_words)
    new_word_data.append(new_sentence)

"""
tfvect.fit_transform(word_data)

fnames = tfvect.get_feature_names()
print len(fnames )
print fnames
print fnames[34597]
""""
print(new_word_data)
print(len(new_word_data))

word_matrix = np.array(new_word_data)
np.reshape(word_matrix,(len(new_word_data),1))
print(word_matrix)
"""

#tfid = TfidfTransformer().fit(word_matrix)
### in Part 4, do TfIdf vectorization here


