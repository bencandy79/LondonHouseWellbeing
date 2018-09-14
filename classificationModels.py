# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:45:53 2018

@author: KU62563
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:13:02 2018

@author: Ben Candy
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import math
from sklearn.metrics import classification_report,confusion_matrix, mean_squared_error, accuracy_score
import statsmodels.api as sm
import seaborn as sns

np.random.seed(100)

# create datasets and specify validation model
dataset = pd.read_csv('files/DataSet.csv')
validation = KFold(n_splits=7, shuffle=True, random_state=None)

columns = ['Education_Employment', 'Safety_Security', 'Environment', 'Vitality_Participation', 
               'Infrastructure', 'Health', 'quintile']
DataVars = pd.DataFrame(dataset, columns=columns)
y = np.ravel(pd.DataFrame(dataset['quintile']))
X = DataVars
X.apply(np.log)
X.replace(np.nan, 0)
del X['quintile']

classification_models = [LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), 
                         RandomForestClassifier(n_estimators=100), GradientBoostingClassifier(), GaussianNB()]
classification_model_labels = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbours',
                               'Random Forest', 'Gradient Boosting', 'Naive Bayes']

result_list = []
# classification models
for i in range(0,len(classification_models)):
    np.random.seed(100)
    classification_score = 0
    training_score = 0
    prediction_score = 0
    for train_index, test_index in validation.split(X):
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        classifier = classification_models[i]  
        model = classifier.fit(X_train, y_train)    
        predictions = model.predict(X_test)
        for j in range(0,len(y_test)):
            prediction_score = prediction_score + ((model.predict(X_test)[j]-y_test[j])*(model.predict(X_test)[j]-y_test[j]))
        training_score = training_score + accuracy_score(y_train, model.predict(X_train))
        classification_score = classification_score + accuracy_score(y_test, predictions)    
    result = [classification_model_labels[i], training_score/7, classification_score/7, math.sqrt(prediction_score/7)]
    result_list.append(result)

result_df = pd.DataFrame(result_list)

result_df.rename(columns={0:'Model Type',1:'Training Score',2:'Test Score',3:'Prediction Error'}, inplace=True)
    

## neural network
#np.random.seed(100)
#for i in range(1,21):
#    classification_score = 0
#    for train_index, test_index in validation.split(X):
#        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
#        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
#        nn = MLPClassifier(hidden_layer_sizes=(i*5),activation='tanh',max_iter=1000)   
#        network = nn.fit(X_train, y_train)    
#        predictions = nn.predict(X_test)
#        classification_score = classification_score + accuracy_score(y_test, predictions)
#    print((i*5),classification_score/7)   
#
#np.random.seed(100)            
#activation_functions = ['identity','logistic','tanh','relu']
#for j in range(0,len(activation_functions)):       
#    classification_score = 0
#    for train_index, test_index in validation.split(X):
#        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
#        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
#        nn = MLPClassifier(hidden_layer_sizes=(55),activation=activation_functions[j],max_iter=1000)   
#        network = nn.fit(X_train, y_train)    
#        predictions = nn.predict(X_test)
#        classification_score = classification_score + accuracy_score(y_test, predictions)
#    print(activation_functions[j],classification_score/7)
#    
    
