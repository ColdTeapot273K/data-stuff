#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'total_stock_value', 'shared_receipt_with_poi',
                 'from_poi_to_this_person'] 
email_features_list = ['to_messages', 'from_poi_to_this_person', 'from_messages',
                       'from_this_person_to_poi', 'shared_receipt_with_poi']
finance_features_list = ['salary', 'bonus', 'total_stock_value']
# feature_1 = "salary"
# feature_2 = "bonus"
# poi = "poi"
# features_list = [poi, feature_1, feature_2]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

'''
# done some EDA to hunt the incorrectly parsed entries:
df = pd.DataFrame.from_dict(data_dict, orient='index')
df[df.columns] = df[df.columns].apply(pd.to_numeric, errors='coerce')
df[df.columns[df.dtypes.eq('float64')]] = df[df.columns[df.dtypes.eq('float64')]].astype('Int64', errors='ignore')
df[df['total_stock_value'] < 0] # only 'BELFER ROBERT' was incorrectly parsed, just drop the guy
'''
data_dict.pop('BELFER ROBERT', 0)

### Task 3: Create new feature(s)
df = pd.DataFrame.from_dict(data_dict, orient='index')
df[df.columns] = df[df.columns].apply(pd.to_numeric, errors='coerce')  # email addressess are misparsed, drop them later
df[df.columns[df.dtypes.eq('float64')]] = df[df.columns[df.dtypes.eq('float64')]].astype('Int64', errors='ignore')

df_modified = df.dropna(subset=email_features_list)
df_modified = df_modified.drop(columns=['deferral_payments', 'loan_advances', 'restricted_stock_deferred',
                                        'deferred_income', 'director_fees']) # drop columns with a lot of NaN
X = df_modified.drop(['poi'], axis=1)
X = X.drop(['email_address'], axis=1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
#imputer = IterativeImputer()
#imputer.fit_transform(X_train)

#X = X.fillna(0) # instead of imputing
from sklearn.impute import SimpleImputer
#imputer = SimpleImputer()

y = df_modified['poi']

# separate financial and mail features to rescale
#X_financial = X.drop(email_features_list, axis=1)
X_financial = X[finance_features_list]
#X_financial['bonus/salary'] = X_financial['bonus'] / X_financial['salary'] # kinda leaky, but imagine we have the data necessary
#X_financial = X_financial.drop(['bonus', 'salary'], axis=1)

#X_financial = X_financial.fillna(0)
imputerF = SimpleImputer(strategy='median')
imputerF.fit(X_financial)
X_financial = imputerF.transform(X_financial)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#scalerF = MinMaxScaler()
scalerF = StandardScaler()
scalerF.fit(X_financial)
X_financial = scalerF.transform(X_financial)
X_financial = pd.DataFrame(X_financial, index=X.index.values)


X_mail = X[email_features_list]
X_mail['from_poi/from'] = X_mail['from_this_person_to_poi'] / X_mail['from_messages']
X_mail['to_poi/to'] = (X_mail['from_poi_to_this_person'] + X_mail['shared_receipt_with_poi']) / X_mail['to_messages']

X_mail = X_mail.drop(['to_messages', 'from_messages', 'from_this_person_to_poi',
                      'from_poi_to_this_person', 'shared_receipt_with_poi'], axis=1)

imputerM = SimpleImputer(strategy='mean')
imputerM.fit(X_mail)
X_mail = imputerM.transform(X_mail)

#scalerM = MinMaxScaler()
#scalerM = StandardScaler(with_mean=False)
#scalerM.fit(X_mail)
#X_mail = scalerM.transform(X_mail)
#scalerM = StandardScaler()
#scalerM.fit(X_mail)
#X_mail = scalerM.transform(X_mail)
X_mail = pd.DataFrame(X_mail, index=X.index.values)

X_ready = X_financial.join(X_mail, lsuffix='financial')
# X_train['bonus/salary'] = df_modified['bonus'] / df_modified['salary'] # non-leaky
# df_modified = df.fillna()
'''
# Legacy:
feature_21 = "bonus/salary"
feature_2 = feature_21
for i in range(len(finance_features)):
    # finance_features[i] = np.append(finance_features[i], finance_features[i][1]/finance_features[i][0])
    bonus = finance_features[i][1]
    stocks = finance_features[i][2]
    print('bonus:', bonus, 'stocks:', stocks, 'i', i)
    print(finance_features[i])
    if bonus > 0:
        finance_features[i][1] = finance_features[i][1] / finance_features[i][0]
        
#########
#feature_21 = "bonus/salary"
#feature_2 = feature_21
for i in range(len(features)):
 #   features[i] = np.append(features[i], features[i][1]/features[i][0])
 #   bonus = features[i][1]
    #print(features[i])
    if features[i][1] >= 0 and features[i][0] > 0:
        features[i][1] = features[i][1] / features[i][0]
#    else:
 #       print(features[i])

#features_list.append('bonus/salary')        
'''
'''
    to_messages = finance_features[i][0]
    from_poi_to_this_person = finance_features[i][1]
    from_messages = finance_features[i][2]
    from_this_person_to_poi = finance_features[i][3]
    if to_messages > 0:
        finance_features[i][0] = from_poi_to_this_person / to_messages
    if from_messages > 0:
        finance_features[i][1] = from_this_person_to_poi / from_messages
    finance_features[i].pop()
    finance_features[i].pop()
'''
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, finance_features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB

# clf = GaussianNB()
# clf = GaussianProcessClassifier()
clf = KNeighborsClassifier(5)
#clf = RandomForestClassifier()
# clf = SVC()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
features_train, features_test, labels_train, labels_test =\
    train_test_split(X_financial, y, test_size=0.3, random_state=42, stratify=y)

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
    print('i, real, pred {}, {:.0f}, {:.0f}'.format(i, labels_test[i], pred[i]))
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

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


