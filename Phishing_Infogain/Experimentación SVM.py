# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

ds = pd.read_csv('Phishing_Infogain.csv')

y = ds.loc[:,'class']

y.replace(to_replace=["benign","phishing"], value=[0,1],inplace=True)

X = ds.loc[:,ds.columns!='class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

clf_svc = SVC(gamma='auto')

scores_acc = cross_val_score(clf_svc, X,y, cv=10, scoring='accuracy')
scores_pre = cross_val_score(clf_svc, X,y, cv=10, scoring='precision')
scores_rec = cross_val_score(clf_svc, X,y, cv=10, scoring='recall')
scores_f1 = cross_val_score(clf_svc, X,y, cv=10, scoring='f1')

print (scores_acc.mean())
print (scores_pre.mean())
print (scores_rec.mean())
print (scores_f1.mean())

clf_svc.fit(X_train,y_train)

y_pred_svc = clf_svc.predict(X_test)

print ("\nLa exactitud de Maquina de Soporte Vectorial es: ", 
       accuracy_score(y_test,y_pred_svc))
print ("La precision de Maquina de Soporte Vectorial es: ", 
       precision_score(y_test,y_pred_svc))
print ("La recuperacion de Maquina de Soporte Vectorial es: ", 
       recall_score(y_test,y_pred_svc))
print ("El valor F de Maquina de Soporte Vectorial es: ", 
       f1_score(y_test,y_pred_svc))

