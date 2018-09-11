# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:13:02 2018

@author: Ben Candy
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,confusion_matrix, mean_squared_error
import statsmodels.api as sm
import seaborn as sns

# create datasets and specify validation model
dataset = pd.read_csv('files/DataSet.csv')
validation = KFold(n_splits=7, shuffle=True, random_state=None)

columns = ['Education_Employment', 'Safety_Security', 'Environment', 'Vitality_Participation', 
               'Infrastructure', 'Health', 'thousands']
DataVars = pd.DataFrame(dataset, columns=columns)
y = np.ravel(pd.DataFrame(dataset['thousands']))
X = DataVars
del X['thousands']


# use statsmodel for model summary
X_sm = sm.add_constant(X)
stats_model = sm.OLS(y, X_sm)
stats_model_estimate = stats_model.fit()
print(stats_model_estimate.summary())


# linear regression
linearr_score = 0
linearr_error = 0

for train_index, test_index in validation.split(X):
    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index] 
    lm = linear_model.LinearRegression()    
    model = lm.fit(X_train, y_train)    
    predictions = lm.predict(X_test)
    linearr_score = linearr_score + model.score(X_test, y_test)
    linearr_error = linearr_error + mean_squared_error(y_test, predictions)
linearr_score = linearr_score/7
linearr_error = linearr_error/7

lr_list = pd.DataFrame([['Linear Regression', linearr_score, linearr_error]])

linearr_score = 0
linearr_error = 0

poly2 = PolynomialFeatures(degree=2)
X2 = poly2.fit_transform(X)
for train_index, test_index in validation.split(X2):
    X2_train, X2_test = np.array(X2)[train_index], np.array(X2)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index] 
    clm = linear_model.LinearRegression()    
    model = clm.fit(X2_train, y_train)    
    predictions = clm.predict(X2_test)
    linearr_score = linearr_score + model.score(X2_test, y_test)
    linearr_error = linearr_error + mean_squared_error(y_test, predictions)
linearr_score = linearr_score/7
linearr_error = linearr_error/7

lr_list_poly2 = pd.DataFrame([['Linear Regression (poly 2)', linearr_score, linearr_error]])

# reduce features 
del X['Vitality_Participation']
linearr_score = 0
linearr_error = 0

for train_index, test_index in validation.split(X):
    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index] 
    lm = linear_model.LinearRegression()    
    model = lm.fit(X_train, y_train)    
    predictions = lm.predict(X_test)
    linearr_score = linearr_score + model.score(X_test, y_test)
    linearr_error = linearr_error + mean_squared_error(y_test, predictions)
linearr_score = linearr_score/7
linearr_error = linearr_error/7

lr_list_5dim = pd.DataFrame([['Linear Regression (reduced to 5 dim)', linearr_score, linearr_error]])

del X['Education_Employment']
linearr_score = 0
linearr_error = 0

for train_index, test_index in validation.split(X):
    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index] 
    lm = linear_model.LinearRegression()    
    model = lm.fit(X_train, y_train)    
    predictions = lm.predict(X_test)
    linearr_score = linearr_score + model.score(X_test, y_test)
    linearr_error = linearr_error + mean_squared_error(y_test, predictions)
linearr_score = linearr_score/7
linearr_error = linearr_error/7

lr_list_4dim = pd.DataFrame([['Linear Regression (reduced to 4 dim)', linearr_score, linearr_error]])

# try a log transformation of the data
dataset = pd.read_csv('files/DataSet.csv')
validation = KFold(n_splits=7, shuffle=True, random_state=None)

columns = ['Education_Employment', 'Safety_Security', 'Environment', 'Vitality_Participation', 
               'Infrastructure', 'Health', 'thousands']
DataVars = pd.DataFrame(dataset, columns=columns)
y = np.log(np.ravel(pd.DataFrame(dataset['thousands'])))
X = DataVars
X.apply(np.log)
X.replace(np.nan,0)
del X['thousands']

for train_index, test_index in validation.split(X):
    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index] 
    lm = linear_model.LinearRegression()    
    model = lm.fit(X_train, y_train)    
    predictions = lm.predict(X_test)
    linearr_score = linearr_score + model.score(X_test, y_test)
    linearr_error = linearr_error + mean_squared_error(y_test, predictions)
linearr_score = linearr_score/7
linearr_error = linearr_error/7

lr_list_logtrans = pd.DataFrame([['Linear Regression (Log Transform)', linearr_score, linearr_error]])

model_results = pd.concat([lr_list, lr_list_poly2, lr_list_5dim, lr_list_4dim, lr_list_logtrans])
model_results.rename(columns={0:'Model Type',1:'Av. R sq',2:'Av. MSE'},inplace=True)

new_data = pd.DataFrame(dataset)
new_data['predicted'] = np.exp(pd.Series(lm.predict(X)))

model_results.to_csv('files/regression.csv')