# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

ds = pd.read_csv('Phishing_BestFirst.csv')

y = ds.loc[:,'class']

y.replace(to_replace=["benign","phishing"], value=[0,1],inplace=True)

X = ds.loc[:,ds.columns!='class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

clf = MLPClassifier()

scores_acc = cross_val_score(clf, X,y, cv=10, scoring='accuracy')
scores_pre = cross_val_score(clf, X,y, cv=10, scoring='precision')
scores_rec = cross_val_score(clf, X,y, cv=10, scoring='recall')
scores_f1 = cross_val_score(clf, X,y, cv=10, scoring='f1')

print (scores_acc.mean())
print (scores_pre.mean())
print (scores_rec.mean())
print (scores_f1.mean())
    
clf.fit(X_train,y_train)

y_pred_mlp = clf.predict(X_test)

print ("\nLa exactitud de Redes Neuronales con Perceptron Multicapas es:  ", 
       accuracy_score(y_test,y_pred_mlp))
print ("La precision de Redes Neuronales con Perceptron Multicapas es: ", 
       precision_score(y_test,y_pred_mlp))
print ("La recuperacion de Redes Neuronales con Perceptron Multicapas es: ", 
       recall_score(y_test,y_pred_mlp))
print ("El valor F de Redes Neuronales con Perceptron Multicapas es: ", 
       f1_score(y_test,y_pred_mlp))

"""parameters = {'activation':["identity", "logistic", "tanh", "relu"],
              'solver': ["lbfgs","sgd","adam"],
              'learning_rate': ["constant","invscaling","adaptive"],
              'shuffle': [True,False]}

grid_clf = GridSearchCV(clf,parameters,cv=10)

grid_clf.fit(X_train,y_train)

print (grid_clf.best_score_)

print (grid_clf.best_estimator_)

print (grid_clf.best_params_)
"""