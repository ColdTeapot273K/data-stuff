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

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys='../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)



### your code goes here 

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)
print("accuracy:", accuracy)

poi_q = 0
for i in labels_test:
    poi_q += i

print('poi_q', poi_q, 'people_q', len(labels_test))

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
for i in range(len(labels_test)):
    print('i, real, pred', i, labels_test[i], pred[i])
    if labels_test[i] == pred[i] and labels_test[i] == 1:
        true_pos += 1
    if labels_test[i] == pred[i] and labels_test[i] == 0:
        true_neg += 1
    if labels_test[i] != pred[i] and labels_test[i] == 0:
        false_pos += 1
    if labels_test[i] != pred[i] and labels_test[i] == 1:
        false_neg += 1

print('true_pos, true_neg, false_pos, false_neg', true_pos, true_neg, false_pos, false_neg)
print('precision, recall', precision_score(labels_test, pred), recall_score(labels_test, pred))
