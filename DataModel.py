# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 18:15:40 2018
@author: Ben Candy
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns

class DataModel:
    dataset = pd.read_csv('files/DataSet.csv')
    validation = KFold(n_splits=7, shuffle=True, random_state=None)
  
    def __init__(self, model_type):
        if model_type == 'regression':
            self._model_type = 'regression'
            self._Y = 'Median_Price'
        elif model_type == 'classification':
            self._Y = 'quintile'
            self._model_type = 'classification'
        else:
            print('Please choose either regression or classification')
        self._y = np.ravel(pd.DataFrame(self.dataset[self._Y]))
        self._columns = ['Education_Employment', 'Safety_Security', 'Environment', 'Vitality_Participation', 
               'Infrastructure', 'Health', self._Y]
        self._DataVars = pd.DataFrame(self.dataset, columns=self._columns)
        self._x = self._DataVars.drop(self._Y)
        self._linearr_score = 0
        self._logr_score = 0
        self._regtree_score = 0
        self._nn_score = 0

    def pairplot(self):
        if self._model_type == 'regression':    
            p = sns.pairplot(self._DataVars)
        elif self._model_type == 'classification':
            p = sns.pairplot(self._DataVars, hue='quintile')
        p.savefig(self._model_type + '_pairplot.png')    
    
    def correlationMatrix(self):
        return(self._DataVars.cov())
        
    def covarianceMatrix(self):
        return(self._DataVars.cor())
    
    def bestModel(self):
        # regression models
        if self._model_type == 'regression':
            #linear_regression
            for train_index, test_index in self.validation.split(self._x):
                X_train, X_test = np.array(self._x)[train_index], np.array(self._x)[test_index]
                y_train, y_test = np.array(self._y)[train_index], np.array(self._y)[test_index] 
                lm = linear_model.LinearRegression()    
                model = lm.fit(X_train, y_train)    
                predictions = lm.predict(X_test)
                self._linearr_score = self._linearr_score + model.score(X_test, y_test)    
            self._linearr_score = self._linearr_score/7
        #classification models  
        elif self._model_type == 'classification':
            # neural network
            for train_index, test_index in self.validation.split(self._x):
                X_train, X_test = np.array(self._x)[train_index], np.array(self._x)[test_index]
                y_train, y_test = np.array(self._y)[train_index], np.array(self._y)[test_index] 
                nn = MLPClassifier(hidden_layer_sizes=(629,1000,629),max_iter=500)   
                network = nn.fit(X_train, y_train)    
                predictions = nn.predict(X_test)
                print(confusion_matrix(y_test,predictions))    
                print(classification_report(y_test,predictions))
            
    

new_Model = DataModel('regression')

testframe = new_Model.test()




