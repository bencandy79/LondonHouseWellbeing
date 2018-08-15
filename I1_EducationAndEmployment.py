# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 19:51:56 2018

@author: ben.candy
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

STscaler = StandardScaler()
NMscaler = MinMaxScaler()

XLSX_WARD_ATLAS = 'https://files.datapress.com/london/dataset/ward-profiles-and-atlas/2015-09-24T14:48:35/ward-atlas-data.xls'
CSV_SCHOOLS = 'https://files.datapress.com/london/dataset/london-schools-atlas/2017-01-26T18:50:00/all_schools_xy_2016.csv'

# import ward atlas sheet 2
try:
    xlsx_ward_atlas = pd.ExcelFile(XLSX_WARD_ATLAS)
    df_ward_atlas2 = pd.read_excel(XLSX_WARD_ATLAS, 'iadatasheet2', skiprows=2)   
except:
    print('Cannot open data file - Ward Atlas Sheet 2') 

employment_columns = ['New Code','Economically active: % In employment']
df_employment = pd.DataFrame(df_ward_atlas2, columns = employment_columns)
df_employment.rename(columns={'New Code':'Ward_Code','Economically active: % In employment':'Employ_Rate'},inplace=True)
df_employment.set_index('Ward_Code',inplace=True)
df_employment = df_employment.drop(['K04000001'])
employment_scaled_values = STscaler.fit_transform(df_employment)
df_employment.loc[:,:] = employment_scaled_values

# import ward atlas sheet 3
try:
    df_ward_atlas3 = pd.read_excel(XLSX_WARD_ATLAS, 'iadatasheet3', skiprows=2)   
except:
    print('Cannot open data file - Ward Atlas Sheet 3')
    
GCSE_columns = ['New Code','2014.2']
df_GCSE = pd.DataFrame(df_ward_atlas3, columns = GCSE_columns)
df_GCSE.rename(columns={'New Code':'Ward_Code','2014.2':'Av_GCSE_Points'},inplace=True)
df_GCSE.set_index('Ward_Code',inplace=True)
df_GCSE = df_GCSE.drop(['K04000001'])
GCSE_scaled_values = STscaler.fit_transform(df_GCSE)
df_GCSE.loc[:,:] = GCSE_scaled_values


