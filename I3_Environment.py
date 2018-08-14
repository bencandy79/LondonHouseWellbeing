# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 20:48:27 2018

@author: ben.candy
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

STscaler = StandardScaler()
NMscaler = MinMaxScaler()

XLSX_WARD_ATLAS = 'https://files.datapress.com/london/dataset/ward-profiles-and-atlas/2015-09-24T14:48:35/ward-atlas-data.xls'
CSV_NATURE_ACCESS = 'https://files.datapress.com/london/dataset/access-public-open-space-and-nature-ward/public-open-space-nature-ward_access-to-nature.csv'

try:
    xlsx_ward_atlas = pd.ExcelFile(XLSX_WARD_ATLAS)
    df_ward_atlas5 = pd.read_excel(XLSX_WARD_ATLAS, 'iadatasheet5', skiprows=2)   
except:
    print('Cannot open data file - Ward Atlas Sheet 2') 

emissions_columns = ['New Code','2011.4','2011.5','2011.6']
df_emissions = pd.DataFrame(df_ward_atlas5, columns = emissions_columns)
df_emissions.rename(columns={'New Code':'Ward_Code','2011.4':'PM10','2011.5':'NOx','2011.6':'NO2'},inplace=True)
df_emissions.set_index('Ward_Code',inplace=True)
df_emissions = df_emissions.drop(['K04000001'])
df_emissions = df_emissions.drop(['E92000001'])
df_emissions = df_emissions.drop(['E12000007'])

emissions_scaled_values = STscaler.fit_transform(df_emissions)
df_emissions.loc[:,:] = (emissions_scaled_values*-1)
df_emissions['emission'] = (df_emissions['PM10'] + df_emissions['NOx'] + df_emissions['NO2'])/3
df_emissions = pd.DataFrame(df_emissions['emission'])

greenspace_columns = ['New Code','2014.4']
df_greenspace = pd.DataFrame(df_ward_atlas5, columns = greenspace_columns)
df_greenspace.rename(columns={'New Code':'Ward_Code','2014.4':'Greenspace%'},inplace=True)
df_greenspace.set_index('Ward_Code',inplace=True)
df_greenspace = df_greenspace.drop(['K04000001'])
df_greenspace = df_greenspace.drop(['E92000001'])
df_greenspace = df_greenspace.drop(['E12000007'])

greenspace_scaled_values = STscaler.fit_transform(df_greenspace)
df_greenspace.loc[:,:] = greenspace_scaled_values

df_green = pd.merge(df_emissions,df_greenspace, left_index=True, right_index=True)

# import open space and nature access data
try:
    df_nature_access = pd.read_csv(CSV_NATURE_ACCESS)
except:
    print('Cannot open data file - Nature Access by Ward')

nature_columns = ['Ward','% homes with good access to nature']
df_nature = pd.DataFrame(df_nature_access, columns = nature_columns)
df_nature.rename(columns={'Ward':'Ward_Code', '% homes with good access to nature':'Nature_Access'},inplace=True)
df_nature.set_index('Ward_Code',inplace=True)

nature_scaled_values = STscaler.fit_transform(df_nature)
df_nature.loc[:,:] = nature_scaled_values

df_environment = pd.merge(df_green,df_nature, left_index=True, right_index=True)

df_environment_cov = df_environment.cov()
df_environment_corr = df_environment.corr()

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df_environment)
df_environment['Indicator'] = pd.Series(principalComponents[:,0],index=df_environment.index)
normComponents = NMscaler.fit_transform(df_environment)
df_environment_norm = df_environment
df_environment_norm.loc[:,:] = normComponents
indicator_three_environment = df_environment_norm.loc[:,'Indicator']