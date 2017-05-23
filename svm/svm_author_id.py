#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
t0 = time()
from sklearn.svm import SVC
xt = features_train
yt = labels_train
xs = features_test
ys = labels_test
'''
#for i in xt[0:10]:
    #print i
'''
from sklearn.naive_bayes import GaussianNB
clf = SVC(kernel='linear')
clf.fit(xt, yt)
'''
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    #max_iter=-1, probability=False, random_state=None, shrinking=True,
    #tol=0.001, verbose=False)
'''
pred = clf.predict(xs)

#check accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(pred,ys)
print score

#print total time
print "Total  time(s):", round(time()-t0, 3), "s"
print "Total  time(m):", round(time()-t0, 3)/60, "m"

#########################################################
### Answer ###
'''
no. of Chris training emails: 7936
no. of Sara training emails: 7884
0.984072810011
Total  time(s): 161.095 s
Total  time(m): 2.68491666667 m
'''
#########################################################
