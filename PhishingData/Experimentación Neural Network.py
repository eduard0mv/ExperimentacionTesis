# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict

ds = pd.read_csv('PhishingData.csv')

y = ds.loc[:,'Result']

X = ds.loc[:,ds.columns!='Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

clf = MLPClassifier(max_iter=1000)
    
scores_acc = cross_val_score(clf, X,y, cv=10, scoring='accuracy')
scores_pre = cross_val_score(clf, X,y, cv=10, scoring='precision_weighted')
scores_rec = cross_val_score(clf, X,y, cv=10, scoring='recall_weighted')
scores_f1 = cross_val_score(clf, X,y, cv=10, scoring='f1_weighted')

print (scores_acc.mean())
print (scores_pre.mean())
print (scores_rec.mean())
print (scores_f1.mean())

clf.fit(X_train,y_train)

y_pred_mlp = clf.predict(X_test)

print ("\nLa exactitud de Redes Neuronales con Perceptron Multicapas es:  ", 
       accuracy_score(y_test,y_pred_mlp))
print ("La precision de Redes Neuronales con Perceptron Multicapas es: ", 
       precision_score(y_test,y_pred_mlp,average='weighted'))
print ("La recuperacion de Redes Neuronales con Perceptron Multicapas es: ", 
       recall_score(y_test,y_pred_mlp,average='weighted'))
print ("El valor F de Redes Neuronales con Perceptron Multicapas es: ", 
       f1_score(y_test,y_pred_mlp,average='weighted'))
