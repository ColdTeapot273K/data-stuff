#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
#plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
#plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
#plt.legend()
#plt.xlabel("bumpiness")
#plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary
#clf = GaussianNB()
#clf = RandomForestClassifier(n_estimators=500, min_samples_split=10, min_impurity_decrease=0.05, n_jobs=-1)
# good score:
clf = GaussianProcessClassifier(1.0 * RBF(1.0)) #not  the highest score, but good generalization
#clf = KNeighborsClassifier(4) #highest score, but overfits

t0 = time()
#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print("prediction time:", round(time()-t1, 3), "s")

#########################################################
accuracy = clf.score(features_test, labels_test)
print("accuracy:", accuracy)


try:
    prettyPicture(clf, features_test, labels_test)
    plt.show()
except NameError:
    pass
