# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:13:02 2018

@author: Ben Candy
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import powerlaw
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import math
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score


np.random.seed(100)

# create datasets and specify validation model
dataset = pd.read_csv('datastore/DataSet.csv')
validation = KFold(n_splits=7, shuffle=True, random_state=None)

columns = ['Education_Employment', 'Safety_Security', 'Environment', 'Vitality_Participation', 
               'Infrastructure', 'Health', 'thousands']
DataVars = pd.DataFrame(dataset, columns=columns)
DataVars_corr = DataVars.corr()
DataVars_corr.to_csv('datastore/datasetcorrelations.csv')
y = np.ravel(pd.DataFrame(dataset['thousands']))
X = DataVars
del X['thousands']

pl_fit = powerlaw.Fit(y)

pl_alpha = pl_fit.power_law.alpha
pl_sigma = pl_fit.power_law.sigma


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
dataset = pd.read_csv('datastore/DataSet.csv')
validation = KFold(n_splits=7, shuffle=True, random_state=None)

columns = ['Education_Employment', 'Safety_Security', 'Environment', 'Vitality_Participation', 
               'Infrastructure', 'Health', 'Median_Price']
DataVars = pd.DataFrame(dataset, columns=columns)
y = np.log(np.ravel(pd.DataFrame(dataset['Median_Price'])))
X = DataVars
#X.apply(np.log)
#X.replace(np.nan,0)
del X['Median_Price']

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

print(model.coef_, model.intercept_)

new_data = pd.DataFrame(dataset)
new_data['predicted'] = np.exp(pd.Series(lm.predict(X)))

model_results.to_csv('datastore/regression.csv', index=False)

np.random.seed(100)
# create datasets and specify validation model
dataset = pd.read_csv('datastore/DataSet.csv')
validation = KFold(n_splits=7, shuffle=True, random_state=None)

columns = ['Education_Employment', 'Safety_Security', 'Environment', 'Vitality_Participation', 
               'Infrastructure', 'Health', 'quintile']

DataVars = pd.DataFrame(dataset, columns=columns)
y = np.ravel(pd.DataFrame(dataset['quintile']))
X = DataVars
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
result_df.to_csv('datastore/classificationResults.csv', index=False)    

# tune Neural Network for hidden layer
np.random.seed(100)
neural_size_results = []
for i in range(1,21):
    classification_score = 0
    for train_index, test_index in validation.split(X):
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        nn = MLPClassifier(hidden_layer_sizes=(i*5),activation='logistic',max_iter=1000)   
        network = nn.fit(X_train, y_train)    
        predictions = nn.predict(X_test)
        classification_score = classification_score + accuracy_score(y_test, predictions)
    neural_size = [i*5,classification_score/7]
    neural_size_results.append(neural_size)
neuralsize_df = pd.DataFrame(neural_size_results)
neuralsize_df.rename(columns={0:'Nodes in hidden layer',1:'Test Score'}, inplace=True)
neuralsize_df.to_csv('datastore/neuralsizeResults.csv', index=False)   

# tune Neural Network for best activation function
np.random.seed(100)            
activation_functions = ['identity','logistic','tanh','relu']
neural_act_results = []
for j in range(0,len(activation_functions)):       
    classification_score = 0
    for train_index, test_index in validation.split(X):
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        nn = MLPClassifier(hidden_layer_sizes=(45),activation=activation_functions[j],max_iter=1000)   
        network = nn.fit(X_train, y_train)    
        predictions = nn.predict(X_test)
        classification_score = classification_score + accuracy_score(y_test, predictions)
    neural_act = [activation_functions[j],classification_score/7]
    neural_act_results.append(neural_act)
neuralact_df = pd.DataFrame(neural_act_results)
neuralact_df.rename(columns={0:'Activation Function',1:'Test Score'}, inplace=True)
neuralact_df.to_csv('datastore/neuralactivationResults.csv', index=False) 
    
chosen_classifier = LogisticRegression()  
chosen_model = chosen_classifier.fit(X, y)    
chosen_predictions = chosen_model.predict(X)
new_data['Logistic'] = pd.Series(chosen_predictions)
    
neural_classifier = MLPClassifier(hidden_layer_sizes=(45),activation='logistic',max_iter=1000) 
neural_model = neural_classifier.fit(X, y)    
nn_predictions = neural_model.predict(X)
new_data['NeuralNetwork'] = pd.Series(nn_predictions)

new_data.to_csv('datastore/predictions.csv')
