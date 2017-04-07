#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from sklearn import tree
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split


data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

print clf.score(features_test, labels_test)
print sum(clf.predict(features_test))
print len(labels_test)

## Number of true positives
counter = 0
pred = clf.predict(features_test)
actual = labels_test

for i in range(len(pred)):
    if pred[i] == actual[i] ==1:
        counter +=1
print counter

## precision and recall
from sklearn import metrics

precision = metrics.precision_score(actual, pred)
recall = metrics.recall_score(actual, pred)
print recall

