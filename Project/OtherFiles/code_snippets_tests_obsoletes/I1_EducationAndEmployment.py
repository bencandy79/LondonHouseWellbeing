# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:39:02 2018

@author: ku62563
"""

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
CSV_SCHOOLS = 'http://ea-edubase-api-prod.azurewebsites.net/edubase/edubasealldata20180817.csv'

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

# import london schools data
try:
    iter_csv = pd.read_csv(CSV_SCHOOLS, iterator=True, chunksize=1000)
    df_schoolsall = pd.concat([chunk[chunk['GOR (name)'] == 'London'] for chunk in iter_csv])
except:
    print('Cannot open data file - SchoolsData')
    
df_schoolslondon = df_schoolsall[(df_schoolsall['StatutoryLowAge'] <16) & (df_schoolsall['StatutoryHighAge'] > 10)]
df_schoolslondonopen = df_schoolslondon[(df_schoolslondon['SchoolCapacity'] > 0) & (df_schoolslondon['EstablishmentStatus (code)'] == 1)]
df_schools = df_schoolslondonopen.pivot_table(values='SchoolCapacity', index='AdministrativeWard (code)', fill_value=0, aggfunc='sum')

schools_scaled_values = STscaler.fit_transform(df_schools)
df_schools.loc[:,:] = schools_scaled_values

df_edu = pd.merge(df_GCSE,df_schools, left_index=True, right_index=True)
df_eduemploy = pd.merge(df_employment,df_edu, left_index=True, right_index=True)

df_eduemploy_cov = df_eduemploy.cov()
df_eduemploy_corr = df_eduemploy.corr()

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df_eduemploy)
df_eduemploy['Indicator'] = pd.Series(principalComponents[:,0],index=df_eduemploy.index)
normComponents = NMscaler.fit_transform(df_eduemploy)
df_eduemploy_norm = df_eduemploy
df_eduemploy_norm.loc[:,:] = normComponents
indicator_one_eduemploy = df_eduemploy_norm.loc[:,'Indicator']
