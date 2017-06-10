#!/usr/bin/python
import random


def makeTerrainData(n_points=1000):
###############################################################################
### make the toy dataset
    random.seed(42)
    grade = [random.random() for ii in range(0,n_points)]
    bumpy = [random.random() for ii in range(0,n_points)]
    error = [random.random() for ii in range(0,n_points)]
    y = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(0,n_points)]
    for ii in range(0, len(y)):
        if grade[ii]>0.8 or bumpy[ii]>0.8:
            y[ii] = 1.0

### split into train/test sets
    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.75*n_points)
    X_train = X[0:split]
    X_test  = X[split:]
    y_train = y[0:split]
    y_test  = y[split:]

    grade_sig = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==0]
    bumpy_sig = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==0]
    grade_bkg = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==1]
    bumpy_bkg = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==1]

    training_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
            , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}


    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    test_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
            , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}

    return X_train, y_train, X_test, y_test
X_train, y_train, X_test, y_test = makeTerrainData(n_points=1000)

##############################################################
## explore data ##
from sklearn.metrics import accuracy_score
xt = X_train
yt = y_train
xs = X_test
ys = y_test
def print_train():
    count = 0
    for i,e in zip(xt,yt):
        count +=1
        # print count,i,e
    return 'number of x_train '+str(count)

def print_test():
    count = 0
    for i,e in zip(xs,ys):
        count +=1
        # print count,i,e
    return 'number of y_train '+str(count)
print print_train()        
print print_test()
##############################################################
## naive bayes ##

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(xt, yt)
# pred = clf.predict(xs)
# score = accuracy_score(pred,ys)
# print score

## 0.884

##############################################################
## decision tree ##

# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# clf.fit(xt, yt)
# pred = clf.predict(xs)
# score = accuracy_score(pred,ys)
# print score

## 0.908

##############################################################
## KNeighborsClassifier ##

# from sklearn.neighbors import KNeighborsClassifier
# import numpy as np
# clf = KNeighborsClassifier(n_neighbors=3)
# clf.fit(xt, yt)
# pred = clf.predict(xs)
# score = accuracy_score(pred,ys)
# print score

## 0.936

##############################################################
## adaboost ##

# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier()
# clf.fit(xt, yt)
# pred = clf.predict(xs)
# score = accuracy_score(pred,ys)
# print score   

##0.924

##############################################################
## test class ##
# from sklearn.naive_bayes import GaussianNB
# from sklearn import tree
# from sklearn.metrics import accuracy_score
# class data:
#     def __init__(self):
#         self.xt = X_train
#         self.yt = y_train
#         self.xs = X_test
#         self.ys = y_test
#     def use_method(self,method_name):
#         self.method_name = method_name
#         clf = xx()
#         clf.fit(self.xt,self.yt)
#         pred = clf.predict(xs)
#         score = accuracy_score(pred,ys)
#         return score        
# na = self.method_name('GaussianNB')
# dt = self.method_name('tree')

# print na
# print dt

# class method(self,ml_method):
#     self.method[ml_method] = ml_method
#     def naive(self):
#         from sklearn.naive_bayes import GaussianNB
#         clf = GaussianNB()
#         clf.fit(self.xt, self.yt)
#         return score
#     def dt(self):
#         from sklearn import tree
#         clf = tree.DecisionTreeClassifier()
#         clf.fit(self.xt, self.yt)
#         return score
# n = naive()
# d = dt()

# print n.
# print naive(self)
# print dt(self)

##############################################################

## test ##

# lists = [clf1,clf2,clf3]

# for i in lists:
#     pred = i.predict(x_test)
#     print accuracy_score(y_test,pred)


