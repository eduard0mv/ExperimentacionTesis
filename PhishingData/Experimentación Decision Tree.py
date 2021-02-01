# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict

ds = pd.read_csv('PhishingData.csv')

y = ds.loc[:,'Result']

X = ds.loc[:,ds.columns!='Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

clf = DecisionTreeClassifier()

scores_acc = cross_val_score(clf, X,y, cv=10, scoring='accuracy')
scores_pre = cross_val_score(clf, X,y, cv=10, scoring='precision_weighted')
scores_rec = cross_val_score(clf, X,y, cv=10, scoring='recall_weighted')
scores_f1 = cross_val_score(clf, X,y, cv=10, scoring='f1_weighted')

print (scores_acc.mean())
print (scores_pre.mean())
print (scores_rec.mean())
print (scores_f1.mean())

clf.fit(X_train,y_train)

y_pred_dt = clf.predict(X_test)

print ("\nLa exactitud de Maquina de Soporte Vectorial es: ", 
       accuracy_score(y_test,y_pred_dt))
print ("La precision de Maquina de Soporte Vectorial es: ", 
       precision_score(y_test,y_pred_dt,average='weighted'))
print ("La recuperacion de Maquina de Soporte Vectorial es: ", 
       recall_score(y_test,y_pred_dt,average='weighted'))
print ("El valor F de Maquina de Soporte Vectorial es: ", 
       f1_score(y_test,y_pred_dt,average='weighted'))

parameters = {'criterion':["gini","entropy"],
              'max_features': ["auto",None]}

grid_clf = GridSearchCV(clf,parameters,cv=10)

grid_clf.fit(X_train,y_train)

estimator = grid_clf.best_estimator_

y_pred = estimator.predict(X_test)

print (accuracy_score(y_test,y_pred))
print (precision_score(y_test,y_pred,average='weighted'))
print (recall_score(y_test,y_pred,average='weighted'))
print (f1_score(y_test,y_pred,average='weighted'))

#print (grid_clf.best_score_)

print (grid_clf.best_estimator_)

print (grid_clf.best_params_)
