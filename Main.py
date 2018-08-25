# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:39:02 2018
@author: Ben Candy
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

NMscaler = MinMaxScaler()

def standardise(df):
    STscaler = StandardScaler()
    scaled_values = STscaler.fit_transform(df)
    df.loc[:,:] = scaled_values
    return(df)
    
def tablestyle(df):
    th = [('font-size','11px'),('text-align','centre'),('font-weight','bold')
    ,('color','#6d6d6d'),('background-color','#f7f7f9')] 
    td = [('font-size','11px')]
    styles = [dict(selector='th',props=th),dict(selector='td',props=td)]
    df.style.set_table_styles(styles)
    return(df)
    
def domain_indicator(df):
    NMscaler = MinMaxScaler()
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df)
    df['Indicator'] = pd.Series(principalComponents[:,0],index=df.index)
    normComponents = NMscaler.fit_transform(df)
    df_norm = df
    df_norm.loc[:,:] = normComponents
    indicator = df_norm.loc[:,'Indicator']*100
    return(indicator)

# Domain One - Education and Employment
    
df_employment = pd.read_csv('files/Employment.csv')
df_employment.set_index('Ward_Code',inplace=True)
df_employment = df_employment.drop(['K04000001'])
df_employment = standardise(df_employment)

df_GCSE = pd.read_csv('files/GCSE.csv')
df_GCSE.set_index('Ward_Code',inplace=True)
df_GCSE = df_GCSE.drop(['K04000001'])
df_GCSE = standardise(df_GCSE)

df_schoolslondonopen = pd.read_csv('files/Schools.csv')
df_schoolslondonopen['Ofsted_Rating'] = df_schoolslondonopen['OfstedRating (name)']
ofsted_ratings = {'Ofsted_Rating':{'Outstanding':2,'Good':1,'Requires improvement':-1,'Serious Weaknesses':-2,'Special Measures':-2}}
df_schoolslondonopen.replace(ofsted_ratings,inplace=True)
df_schools = df_schoolslondonopen.pivot_table(values='Ofsted_Rating', index='AdministrativeWard (code)', fill_value=0)
df_schools = standardise(df_schools)

# merge the 3 indicators to create the domain
df_edu = pd.merge(df_GCSE,df_schools, left_index=True, right_index=True)
df_eduemploy = pd.merge(df_employment,df_edu, left_index=True, right_index=True)

# check covriance and correlation matrices
df_eduemploy_cov = df_eduemploy.cov()
df_eduemploy_corr = df_eduemploy.corr()

# run principal component analysis to create domain score
Education_Employment = domain_indicator(df_eduemploy)


# Domain Two - Safety and Security

df_crime = pd.read_csv('files/Crime.csv')
df_crime_pivot = df_crime.pivot_table(index='WardCode', columns='Major Category', values='2017',aggfunc='sum')
person_columns = ['Robbery','Sexual Offences','Violence Against the Person']
area_columns = ['Burglary','Criminal Damage','Drugs','Theft and Handling']
df_crimeperson = pd.DataFrame(df_crime_pivot, columns = person_columns)
df_crimeperson = standardise(df_crimeperson)
df_crimearea = pd.DataFrame(df_crime_pivot, columns = area_columns)
df_crimearea = standardise(df_crimearea)

df_traffic = pd.read_csv('files/Traffic.csv')
df_traffic.set_index('Ward_Code',inplace=True)
df_traffic = standardise(df_traffic)
df_traffic['traffic_fatality_score'] = (df_traffic['Fatal'] + df_traffic['Serious'] + df_traffic['Slight'])/3
del df_traffic['Fatal']
del df_traffic['Serious']
del df_traffic['Slight']

# merge the 3 indicators to create the domain
df_crime = pd.merge(df_crimeperson, df_crimearea, left_index=True, right_index=True)
df_safetysecurity = pd.merge(df_crime,df_traffic, left_index=True, right_index=True)

# check covriance and correlation matrices
df_safetysecurity_cov = df_safetysecurity.cov()
df_safetysecurity_corr = df_safetysecurity.corr()

# run principal component analysis to create domain score
Safety_Security = 100-domain_indicator(df_safetysecurity)


# Domain Three - Environment

df_emissions = pd.read_csv('files/Emissions.csv')
df_emissions.set_index('Ward_Code',inplace=True)
df_emissions = df_emissions.drop(['K04000001'])
df_emissions = df_emissions.drop(['E92000001'])
df_emissions = df_emissions.drop(['E12000007'])
df_emissions = standardise(df_emissions)
df_emissions['emission'] = (df_emissions['PM10'] + df_emissions['NO2'])/2
df_emissions = pd.DataFrame(df_emissions['emission']*-1) # emissions negative so *-1

df_greenspace = pd.read_csv('files/Greenspace.csv')
df_greenspace.set_index('Ward_Code',inplace=True)
df_greenspace = df_greenspace.drop(['K04000001'])
df_greenspace = df_greenspace.drop(['E92000001'])
df_greenspace = df_greenspace.drop(['E12000007'])
df_greenspace = standardise(df_greenspace)

df_nature = pd.read_csv('files/Nature.csv')
df_nature.set_index('Ward_Code',inplace=True)
df_nature = standardise(df_nature)

df_green = pd.merge(df_nature,df_greenspace, left_index=True, right_index=True)
df_environment = pd.merge(df_green,df_emissions, left_index=True, right_index=True)

df_environment_cov = df_environment.cov()
df_environment_corr = df_environment.corr()

Environment = domain_indicator(df_environment)


# Domain Four - Community Vitality and Participation

df_turnout = pd.read_csv('files/ElectionTurnout.csv')
df_turnout.set_index('Ward_Code', inplace=True)
df_turnout = standardise(df_turnout)


# Domain Five - Infrastructure

df_pta = pd.read_csv('files/TransportAccess.csv')
df_pta.set_index('Ward Code',inplace=True)
df_pta = standardise(df_pta)

df_density = pd.read_csv('files/PopDensity.csv')
df_density.set_index('Code',inplace=True)
df_density = standardise(df_density)


# Domain Six - Health

df_expectancy = pd.read_csv('files/LifeExpectancy.csv')
df_expectancy.set_index('Ward_Code', inplace=True)
df_expectancy = standardise(df_expectancy)

