# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 18:15:40 2018

@author: Ben Candy
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report,confusion_matrix
from shapely.geometry import Point, Polygon
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import seaborn as sns

DataSet = pd.read_csv('DataSet.csv')

columns = ['Education_Employment', 'Safety_Security', 'Environment', 'Vitality_Participation', 
           'Infrastructure', 'Health']

DataVars = pd.DataFrame(DataSet, columns=columns)
g = sns.pairplot(DataVars)

data = pd.DataFrame(DataSet, columns = columns)

y = pd.DataFrame(DataSet['Median_Price'])

kf = KFold(n_splits=7, shuffle=True, random_state=None)
av_score = 0

for train_index, test_index in kf.split(data):
    X_train, X_test = np.array(data)[train_index], np.array(data)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index] 
    lm = linear_model.LinearRegression()    
    model = lm.fit(X_train, y_train)    
    predictions = lm.predict(X_test)
    av_score = av_score + model.score(X_test, y_test)    
    #plt.scatter(y_test, predictions)
    #plt.xlabel('Actuals')
    #plt.ylabel('Predictions')
print ('Score: ', av_score/7)

# regression modelling

y = pd.DataFrame(DataSet['quintile'])

for train_index, test_index in kf.split(data):
    X_train, X_test = np.array(data)[train_index], np.array(data)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index] 
    nn = MLPClassifier(hidden_layer_sizes=(629,1000,629),max_iter=500)   
    network = nn.fit(X_train, y_train)    
    predictions = nn.predict(X_test)
    print(confusion_matrix(y_test,predictions))    
    print(classification_report(y_test,predictions))
    
# classification modelling
