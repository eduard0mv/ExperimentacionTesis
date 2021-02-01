# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict

ds = pd.read_csv('Phishing_Legitimate_Full.csv')

y = ds.loc[:,'CLASS_LABEL']

X = ds.loc[:,ds.columns!='CLASS_LABEL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

clf = RandomForestClassifier(n_estimators=100,criterion='gini',bootstrap=False,class_weight='balanced')

scores_acc = cross_val_score(clf, X,y, cv=10, scoring='accuracy')
scores_pre = cross_val_score(clf, X,y, cv=10, scoring='precision')
scores_rec = cross_val_score(clf, X,y, cv=10, scoring='recall')
scores_f1 = cross_val_score(clf, X,y, cv=10, scoring='f1')

print (scores_acc.mean())
print (scores_pre.mean())
print (scores_rec.mean())
print (scores_f1.mean())

clf.fit(X_train,y_train)

y_pred_rf = clf.predict(X_test)

print ("\nLa exactitud de Bosque Aleatorio es: ", 
       accuracy_score(y_test,y_pred_rf))
print ("La precision de Bosque Aleatorio es: ", 
       precision_score(y_test,y_pred_rf))
print ("La recuperacion de Bosque Aleatorio es: ", 
       recall_score(y_test,y_pred_rf))
print ("El valor F de Bosque Aleatorio es: ", 
       f1_score(y_test,y_pred_rf))

parameters = {'bootstrap':[True,False],
              'criterion':['gini','entropy'],
              'class_weight':['balanced',None]}

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
