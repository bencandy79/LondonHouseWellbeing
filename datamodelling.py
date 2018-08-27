# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 18:15:40 2018

@author: Ben Candy
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, KFold
from shapely.geometry import Point, Polygon
from pyproj import Proj, transform
import matplotlib.pyplot as plt


DataSet = pd.read_csv('files/DataSet.csv')

columns = ['Education_Employment', 'Safety_Security', 'Environment', 'Vitality_Participation', 
           'Infrastructure', 'Health']

data = pd.DataFrame(DataSet, columns = columns)

y = pd.DataFrame(DataSet['Median_Price'])

#X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)

kf = KFold(n_splits=7, shuffle=True, random_state=None)
kf.get_n_splits(data)

print(kf)

for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lm = linear_model.LinearRegression()    
    model = lm.fit(X_train, y_train)    
    predictions = lm.predict(X_test)    
    plt.scatter(y_test, predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')    
    print ('Score: ', model.score(X_test, y_test))


# regression modelling

    

# classification modelling
